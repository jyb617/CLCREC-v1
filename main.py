import argparse
import os
import time
import numpy as np
import torch
import random
from Dataset import TrainingDataset, data_load
from model_CLCRec import CLCRec  # 使用修复版模型
from torch.utils.data import DataLoader
from Train import train
from Full_rank import full_ranking
from torch.utils.tensorboard import SummaryWriter


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='improved', help='Filename')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lr_lambda', type=float, default=0.5, help='Weight loss one.')
    parser.add_argument('--reg_weight', type=float, default=1e-1, help='Weight decay.')
    parser.add_argument('--temp_value', type=float, default=2.0, help='Contrastive temp_value.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=128, help='Negative size.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=4, help='Workers number.')
    parser.add_argument('--num_sample', type=float, default=0.5, help='Hybrid sampling ratio.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--topK', type=int, default=10, help='Top-K for evaluation.')
    parser.add_argument('--step', type=int, default=2000, help='Evaluation step.')

    parser.add_argument('--has_v', default='True', help='Has Visual Features.')
    parser.add_argument('--has_a', default='True', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init()

    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 🚀 性能优化：启用 cuDNN benchmark
    # 注意：如果需要完全可重复的结果，可能需要关闭 benchmark
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cudnn.deterministic = False  # 提高性能，但牺牲一些可重复性

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"Using device: {device}")

    # 参数设置
    data_path = args.data_path
    save_file_name = args.save_file
    learning_rate = args.l_r
    lr_lambda = args.lr_lambda
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_neg = args.num_neg
    num_sample = args.num_sample
    topK = args.topK
    temp_value = args.temp_value
    step = args.step
    dim_E = args.dim_E

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False
    is_word = True if data_path == 'tiktok' else False

    writer = SummaryWriter()

    print('=' * 80)
    print('Improved CLCRec with Neighbor-Item Contrastive Learning')
    print('=' * 80)

    # 数据加载
    print('\n[1/4] Loading data...')
    try:
        num_user, num_item, num_warm_item, train_data, val_data, val_warm_data, val_cold_data, \
            test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat = data_load(data_path, has_v, has_a,
                                                                                          has_t)

        dir_str = './Data/' + data_path
        user_item_all_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        user_item_train_dict = np.load(dir_str + '/user_item_train_dict.npy', allow_pickle=True).item()
        warm_item = torch.tensor(np.load(dir_str + '/warm_set.npy'))
        cold_item = torch.tensor(np.load(dir_str + '/cold_set.npy'))

        print(f'✓ Data loaded successfully!')
        print(f'  Users: {num_user}, Items: {num_item}, Warm Items: {num_warm_item}')
        print(f'  Train samples: {len(train_data)}')

    except Exception as e:
        print(f'✗ Error loading data: {e}')
        import traceback

        traceback.print_exc()
        exit(1)

    # 创建数据集
    print('\n[2/4] Creating dataset...')
    try:
        train_dataset = TrainingDataset(num_user, num_item, user_item_all_dict, data_path, train_data, num_neg)
        # 🚀 优化 DataLoader 配置
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # 加速 CPU-GPU 数据传输
            persistent_workers=True  # 保持 worker 进程不被销毁
        )
        print(f'✓ Dataset created successfully!')
    except Exception as e:
        print(f'✗ Error creating dataset: {e}')
        import traceback

        traceback.print_exc()
        exit(1)

    # 创建模型
    print('\n[3/4] Building improved model...')
    try:
        model = CLCRec(num_user, num_item, num_warm_item, train_data, reg_weight, dim_E,
                       v_feat, a_feat, t_feat, temp_value, num_neg, lr_lambda, is_word, num_sample).cuda()

        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])

        print(f'✓ Model created successfully!')
        print(f'  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    except Exception as e:
        print(f'✗ Error creating model: {e}')
        import traceback

        traceback.print_exc()
        exit(1)

    # 训练
    print('\n[4/4] Start training...')
    print('=' * 80)

    max_recall = 0.0
    num_decreases = 0
    best_results = {}

    for epoch in range(num_epoch):
        print(f'\n>>> Epoch {epoch + 1}/{num_epoch}')

        try:
            # 训练
            loss, neighbor_loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size,
                                        writer)

            if torch.isnan(loss):
                print('✗ Training diverged (NaN loss)!')
                break

            torch.cuda.empty_cache()

            # 验证集评估
            val_result = full_ranking(epoch, model, val_data, user_item_train_dict, None, False, step, topK, 'Val',
                                      writer)
            val_result_warm = full_ranking(epoch, model, val_warm_data, user_item_train_dict, cold_item, False, step,
                                           topK,
                                           'Val/warm', writer)
            val_result_cold = full_ranking(epoch, model, val_cold_data, user_item_train_dict, warm_item, False, step,
                                           topK,
                                           'Val/cold', writer)

            # 测试集评估
            test_result = full_ranking(epoch, model, test_data, user_item_train_dict, None, False, step, topK, 'Test',
                                       writer)
            test_result_warm = full_ranking(epoch, model, test_warm_data, user_item_train_dict, cold_item, False, step,
                                            topK, 'Test/warm', writer)
            test_result_cold = full_ranking(epoch, model, test_cold_data, user_item_train_dict, warm_item, False, step,
                                            topK, 'Test/cold', writer)

            # 早停机制
            if val_result[1] > max_recall:
                max_recall = val_result[1]
                best_results = {
                    'epoch': epoch,
                    'val': val_result,
                    'val_warm': val_result_warm,
                    'val_cold': val_result_cold,
                    'test': test_result,
                    'test_warm': test_result_warm,
                    'test_cold': test_result_cold
                }
                num_decreases = 0
                print(f'✓ New best model! (Recall@{topK}: {max_recall:.4f})')
            else:
                num_decreases += 1
                if num_decreases > 5:
                    print('\n' + '=' * 80)
                    print('Early stopping triggered!')
                    break

        except Exception as e:
            print(f'✗ Error during epoch {epoch + 1}: {e}')
            import traceback

            traceback.print_exc()
            # 继续下一个epoch而不是直接退出
            continue

    # 保存最佳结果
    print('\n' + '=' * 80)
    print('BEST RESULTS')
    print('=' * 80)

    if best_results:
        print(f"Best Epoch: {best_results.get('epoch', -1) + 1}")
        print(f"\nValidation:")
        print(
            f"  Full    - P@{topK}: {best_results['val'][0]:.4f}, R@{topK}: {best_results['val'][1]:.4f}, NDCG@{topK}: {best_results['val'][2]:.4f}")
        print(
            f"  Warm    - P@{topK}: {best_results['val_warm'][0]:.4f}, R@{topK}: {best_results['val_warm'][1]:.4f}, NDCG@{topK}: {best_results['val_warm'][2]:.4f}")
        print(
            f"  Cold    - P@{topK}: {best_results['val_cold'][0]:.4f}, R@{topK}: {best_results['val_cold'][1]:.4f}, NDCG@{topK}: {best_results['val_cold'][2]:.4f}")
        print(f"\nTest:")
        print(
            f"  Full    - P@{topK}: {best_results['test'][0]:.4f}, R@{topK}: {best_results['test'][1]:.4f}, NDCG@{topK}: {best_results['test'][2]:.4f}")
        print(
            f"  Warm    - P@{topK}: {best_results['test_warm'][0]:.4f}, R@{topK}: {best_results['test_warm'][1]:.4f}, NDCG@{topK}: {best_results['test_warm'][2]:.4f}")
        print(
            f"  Cold    - P@{topK}: {best_results['test_cold'][0]:.4f}, R@{topK}: {best_results['test_cold'][1]:.4f}, NDCG@{topK}: {best_results['test_cold'][2]:.4f}")
        print('=' * 80)

        # 保存到文件
        result_dir = './Data/' + data_path
        os.makedirs(result_dir, exist_ok=True)

        with open(result_dir + '/result_{0}.txt'.format(save_file_name), 'a') as f:
            f.write(str(args) + '\n')
            f.write(f"Best Epoch: {best_results.get('epoch', -1) + 1}\n")
            f.write(
                f"Val: P@{topK}={best_results['val'][0]:.4f}, R@{topK}={best_results['val'][1]:.4f}, NDCG@{topK}={best_results['val'][2]:.4f}\n")
            f.write(
                f"Test: P@{topK}={best_results['test'][0]:.4f}, R@{topK}={best_results['test'][1]:.4f}, NDCG@{topK}={best_results['test'][2]:.4f}\n")
            f.write(
                f"Test Cold: P@{topK}={best_results['test_cold'][0]:.4f}, R@{topK}={best_results['test_cold'][1]:.4f}, NDCG@{topK}={best_results['test_cold'][2]:.4f}\n")
            f.write('-' * 80 + '\n')

        print(f'\n✓ Results saved to {result_dir}/result_{save_file_name}.txt')
    else:
        print("No valid results obtained during training.")
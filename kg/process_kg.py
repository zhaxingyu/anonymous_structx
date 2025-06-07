import argparse
import os
import sys
from datasets import load_dataset, DatasetDict

# 确保 kg 目录在 Python 的搜索路径中，或者 kg 包已正确安装
# 如果 run_webqsp_processing.py 与 kg 目录在同一级，则可以直接导入
try:
    from kg.webqsp import Constructor
except ImportError:
    # 如果直接导入失败，尝试将 kg 的父目录添加到 sys.path
    # 这取决于你的项目结构和运行方式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    # 再次尝试导入，如果你的 kg 目录不在父目录，你可能需要调整路径
    from kg.webqsp import Constructor


# 用于模拟 webqsp.py 中期望的 args 结构
class ArgsForWebQSP:
    class DatasetArgs:
        def __init__(self, use_cache):
            self.use_cache = use_cache

    def __init__(self, use_cache_flag):
        self.dataset = self.DatasetArgs(use_cache_flag)


def main():
    parser = argparse.ArgumentParser(description="Process WebQSP dataset using kg.webqsp.Constructor")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the directory containing WebQSP JSON files.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the processed data and cache.")
    parser.add_argument("--train_file_name", type=str, default="WebQSP.train.json",
                        help="Name of the training data JSON file.")
    parser.add_argument("--test_file_name", type=str, default="WebQSP.test.json",
                        help="Name of the test data JSON file.")
    parser.add_argument("--validation_file_name", type=str, default="WebQSP.validation.json",
                        help="Name of the validation data JSON file (e.g., WebQSP.validation.json or WebQSP.dev.json).")
    parser.add_argument("--use_cache", action="store_true",
                        help="Enable caching of processed datasets.")
    parser.add_argument("--create_validation_from_train", action="store_true",
                        help="If validation file is not found, create it by splitting the training set.")
    parser.add_argument("--validation_split_size", type=float, default=0.1,
                        help="Proportion of the training set to use for validation if creating validation split (e.g., 0.1 for 10%).")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for splitting train/validation set.")

    cli_args = parser.parse_args()

    # 准备传递给 Constructor 的参数对象
    webqsp_args = ArgsForWebQSP(use_cache_flag=cli_args.use_cache)

    # --- 1. 加载数据集 ---
    data_files = {}
    train_file_path = os.path.join(cli_args.input_path, cli_args.train_file_name)
    test_file_path = os.path.join(cli_args.input_path, cli_args.test_file_name)
    validation_file_path = os.path.join(cli_args.input_path, cli_args.validation_file_name)

    if not os.path.exists(train_file_path):
        print(f"错误：训练文件未找到: {train_file_path}")
        sys.exit(1)
    if not os.path.exists(test_file_path):
        print(f"错误：测试文件未找到: {test_file_path}")
        sys.exit(1)

    data_files['train'] = train_file_path
    data_files['test'] = test_file_path
    raw_datasets = None

    if os.path.exists(validation_file_path):
        print(f"找到验证文件: {validation_file_path}")
        data_files['validation'] = validation_file_path
        try:
            raw_datasets = load_dataset('json', data_files=data_files)
        except Exception as e:
            print(f"加载数据集时出错: {e}")
            sys.exit(1)
    elif cli_args.create_validation_from_train:
        print(f"警告：验证文件 '{validation_file_path}' 未找到。将从训练集中创建验证集。")
        # 先只加载训练集和测试集
        temp_data_files = {'train': train_file_path, 'test': test_file_path}
        try:
            temp_raw_datasets = load_dataset('json', data_files=temp_data_files)
        except Exception as e:
            print(f"加载训练/测试数据集以进行分割时出错: {e}")
            sys.exit(1)

        # 检查训练集大小是否足够分割
        if len(temp_raw_datasets['train']) < 2 or \
                (cli_args.validation_split_size * len(temp_raw_datasets['train'])) < 1:
            print(
                f"错误：训练集中的数据不足 (大小: {len(temp_raw_datasets['train'])}) "
                f"以创建大小为 {cli_args.validation_split_size} 的验证分割。"
                "请考虑使用较小的 validation_split_size 或提供一个验证文件。"
            )
            sys.exit(1)

        # 从训练集中分割出验证集
        # train_test_split 返回一个 DatasetDict，包含 'train' 和 'test' 两个key
        train_val_split = temp_raw_datasets['train'].train_test_split(
            test_size=cli_args.validation_split_size,
            shuffle=True,
            seed=cli_args.random_seed
        )
        raw_datasets = DatasetDict({
            'train': train_val_split['train'],
            'validation': train_val_split['test'],  # train_test_split 生成的 'test' 部分作为我们的验证集
            'test': temp_raw_datasets['test']
        })
        print(f"已从训练集分割出验证集: "
              f"{len(raw_datasets['train'])} 训练样本, "
              f"{len(raw_datasets['validation'])} 验证样本.")
    else:
        print(f"错误：验证文件 '{validation_file_path}' 未找到，并且未选择从训练集创建。")
        print("kg.webqsp.py 中的 Constructor 需要 'train', 'validation', 'test' 三个部分。")
        sys.exit(1)

    if raw_datasets is None or not all(split in raw_datasets for split in ['train', 'validation', 'test']):
        print("错误：未能成功加载或创建所有必需的数据集分割 (train, validation, test)。")
        sys.exit(1)

    print("\n成功加载/准备 DatasetDict，包含以下分割:")
    for split_name, dataset_obj in raw_datasets.items():
        print(f"- {split_name}: {len(dataset_obj)} 样本")
    print("-" * 30)

    # --- 2. 调用 Constructor 进行处理 ---
    try:
        print(f"\n正在使用 Constructor 处理数据集...")
        print(f"输入路径: {cli_args.input_path}")
        print(f"输出/缓存路径: {cli_args.output_path}")
        print(f"使用缓存: {cli_args.use_cache}")

        constructor_instance = Constructor(webqsp_args)
        # 调用 to_seq2seq 方法，它会打印数据集大小并返回处理后的 datasets
        # 如果 Constructor 中的断言失败，这里会抛出 AssertionError
        train_ds, dev_ds, test_ds = constructor_instance.to_seq2seq(raw_datasets, cli_args.output_path)

        print("\n数据集处理成功完成!")
        # 你可以在这里添加更多逻辑，比如保存这些 dataset 对象或进行其他操作

    except FileNotFoundError as e:
        print(f"文件错误: {e}")
    except AssertionError as e:
        print(f"断言错误 (通常意味着数据集分割不符合预期): {e}")
    except Exception as e:
        import traceback
        print(f"处理过程中发生未知错误: {e}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
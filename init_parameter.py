import argparse

def init_model():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='C:/store/pycharm1/ROIC/data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--save_results_path", type=str, default='C:/store/pycharm1/ROIC/outputs', help="the path to save results")

    parser.add_argument("--pretrain_dir", default='models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--bert_model", default="C:/store/pycharm1/ROIC/uncased_L-12_H-768_A-12", type=str,
                        help="The path for the pre-trained bert model.")

    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT")

    parser.add_argument("--save_model", action="store_true", help="save trained-model")

    parser.add_argument("--save_results", action="store_true", help="save test results")

    parser.add_argument("--pretrain_dataset", default=None, type=str, required=True,
                        help="The name of the dataset to use for pretraining")

    parser.add_argument("--train_dataset", default=None, type=str, required=True,
                        help="The name of the dataset to use for training")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True,
                        help="The number of known classes")

    parser.add_argument("--labeled_ratio", default=1.0, type=float, required=True,
                        help="The ratio of labeled samples in the training set")

    parser.add_argument("--method", type=str, default=None, help="which method to use")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--lr", default=2e-5, type=float,
                        help="The learning rate of BERT.")

    parser.add_argument("--num_train_epochs", default=100, type=float,

                        help="Total number of training epochs to perform.")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")

    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=10, type=int,
                        help="Patient steps for Early Stop.")

    parser.add_argument("--lr_boundary", type=float, default=0.05, help="The learning rate of the decision boundary.")

    parser.add_argument("--few", type=bool, default=False, help="Few-shot")

    parser.add_argument("--shot", type=int, default=5, help="Few-shot")

    parser.add_argument("--use_rp", type=bool, default=True, help="Use Random Projection")

    parser.add_argument("--rp_dim", type=int, default=20000, help="The dimension for random projection")

    parser.add_argument("--lambda_ridge", type=float, default=1.0, help="The regularization parameter for ridge regression")

    parser.add_argument("--num_contrastive_epochs", type=int, default=50, help="Number of epochs for contrastive learning.")

    parser.add_argument("--save_model_path", type=str, default="./contrastive_model", help="Path to save the model")

    parser.add_argument('--hidden_size', type=int, default=768, help="Size of the hidden layer in the model")

    return parser
from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from loss import *
import random

class ModelManager:
    def __init__(self, args, data, pretrained_model=None, margin=10.0):
        self.model = pretrained_model

        if self.model is None:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=data.num_labels)
            self.restore_model(args)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None
        self.Wo = None
        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.unseen_token_id = data.unseen_token_id
        self.margin = margin
        self.loss_func = DCLLoss(margin=self.margin)

    def open_classify(self, features):
        h_test = torch.nn.functional.relu(features @ self.W_rand)
        logits = h_test @ self.Wo
        preds = torch.argmax(logits, dim=1)
        euc_dis = torch.norm(h_test - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = self.unseen_token_id
        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                features = pooled_output
                preds = self.open_classify(features)
                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, preds))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['F1-score']
            return eval_score

        elif mode == 'test':
            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc
            self.test_results = results
            self.save_results(args)

    def train(self, args, data):
        self.contrastive_train(args, data)
        self.load_model(args, 'contrastive_model.bin')
        rp_features, all_labels = self.apply_ranpac(args, data.train_dataloader)
        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.rp_dim)
        self.delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr=args.lr_boundary)
        self.centroids = self.calculate_centroids_from_features(rp_features, all_labels, data.num_labels)
        self.count_all_parameters(criterion_boundary)
        wait = 0
        best_delta, best_centroids = None, None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(
                    tqdm(data.step2_train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    features = torch.nn.functional.relu(features @ self.W_rand)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    nb_tr_examples += features.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)
            loss = tr_loss / nb_tr_steps
            print('train_loss', loss)
            eval_score = self.evaluation(args, data, mode="eval")
            print('eval_score', eval_score)

            if eval_score >= self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.delta = best_delta
        self.centroids = best_centroids

    def load_model(self, args, model_name):
        output_model_file = os.path.join(args.save_model_path, model_name)
        self.model.load_state_dict(torch.load(output_model_file))

    def contrastive_train(self, args, data):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        for epoch in trange(int(args.num_contrastive_epochs), desc="Contrastive Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(data.step2_train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss = self.loss_func(features, label_ids)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    print(f'Iteration {step + 1}/{len(data.step2_train_dataloader)} - Loss: {loss.item()}')
            loss = tr_loss / nb_tr_steps
            print(f'Epoch {epoch + 1}/{args.num_contrastive_epochs} - Average Loss: {loss}')
        self.save_model(args, 'contrastive_model.bin')

    def save_model(self, args, model_name):
        save_path = os.path.join(args.save_model_path, model_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def apply_ranpac(self, args, dataloader):
        self.model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                all_features.append(features)
                all_labels.append(label_ids)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        rp_dim = args.rp_dim
        self.W_rand = torch.randn(all_features.size(1), rp_dim).to(self.device)
        rp_features = torch.nn.functional.relu(all_features @ self.W_rand)
        self.centroids = self.calculate_centroids_from_features(rp_features, all_labels, len(set(all_labels.tolist())))
        self.Wo = self.compute_ridge_regression(rp_features, all_labels, len(set(all_labels.tolist())),
                                                args.lambda_ridge)
        return rp_features, all_labels

    def calculate_centroids_from_features(self, features, labels, num_labels):
        centroids = torch.zeros(num_labels, features.size(1)).to(self.device)
        for label in range(num_labels):
            indices = (labels == label).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                centroids[label] = features[indices].mean(dim=0)
        return centroids

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def compute_ridge_regression(self, features, labels, num_labels, lambda_ridge):
        H = features
        Y = F.one_hot(labels, num_classes=num_labels).float().to(self.device)
        G = H.t() @ H
        C = H.t() @ Y
        Wo = torch.linalg.solve(G + lambda_ridge * torch.eye(G.size(0)).to(self.device), C)
        return Wo

    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.train_dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        np.save(os.path.join(args.save_results_path, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        np.save(os.path.join(args.save_results_path, 'deltas.npy'), self.delta.detach().cpu().numpy())

        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            # df1 = df1.append(new, ignore_index=True)
            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)
        print('test_results', data_diagram)

    def count_all_parameters(self, criterion_boundary=None):
        bert_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        boundary_params = 0
        if criterion_boundary is not None:
            boundary_params = sum(p.numel() for p in criterion_boundary.parameters() if p.requires_grad)
        total_params = bert_params + boundary_params
        print(f"BERT Trainable Parameters: {bert_params}")
        print(f"BoundaryLoss Trainable Parameters: {boundary_params}")
        print(f"Total Trainable Parameters: {total_params}")


if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    os.makedirs(args.save_model_path, exist_ok=True)

    print('Pre-training begin...')
    pretrain_data = Data(args, args.pretrain_dataset)
    manager_p = PretrainModelManager(args, pretrain_data)
    manager_p.train(args, pretrain_data)
    print('Pre-training finished!')

    print('Training begin...')
    train_data = Data(args, args.train_dataset)
    manager = ModelManager(args, train_data, manager_p.model)
    manager.train(args, train_data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, train_data, mode="test")
    print('Evaluation finished!')


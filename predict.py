import argparse
import preprocessing
import model
import train
from torch.utils.data import DataLoader
import torch


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

model_weights_path = 'models/model_2020-05-30_21:15_742_0.98.pkl'
net = model.MaskDetector(None)
net.load_state_dict(torch.load(model_weights_path, map_location=lambda storage, loc: storage))
net = model.to_gpu(net)
test_dataset = preprocessing.FaceMaskDataset(args.input_folder, [])
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
prediction_df = train.predict(net, test_loader)
f1, roc_auc = train.calculate_scores(prediction_df)
print('F1:', f1, 'ROC_AUC:', roc_auc)
prediction_df[['id', 'pred']].to_csv("prediction.csv", index=False, header=False)

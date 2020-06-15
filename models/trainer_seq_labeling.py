import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..data.sequence_labeling_data import ELMoData

DATA = pd.read_excel('../data/train.xlsx')
PAIRS = [(text, tags) for text, tags in zip(DATA.text, DATA.tags)]
DATASET = ELMoData(PAIRS)
DEVICE = torch.device('cuda')

TRAIN_DATA, TEST_DATA = random_split(DATASET, [int(0.8 * len(DATASET)), int(0.2 * len(DATASET))])
TRAIN_DATA_LOADER = DataLoader(TRAIN_DATA, batch_size=64)
TEST_DATA_LOADER = DataLoader(TEST_DATA, batch_size=64)


def filter_single_tags(prediction):
    """
    зануляет предсказание, если модель считает, что плеоназм состоит из 1 слова
    """
    res = []
    prediction = prediction.tolist()

    for sentence in prediction:
        if sentence.count(2) == 1:
            for i in range(len(sentence)):
                if sentence[i] == 2:
                    sentence[i] = 1
        res.append(sentence)
    return np.asarray(res)


def train(model, optimizer, criterion, epochs=10, best_test_loss=10,
          train=TRAIN_DATA_LOADER, test=TEST_DATA_LOADER, device=DEVICE):

    losses = []

    for n_epoch in range(epochs):

        train_losses = []
        test_losses = []
        test_preds = []
        test_targets = []

        progress_bar = tqdm(total=len(train.dataset), desc='Epoch {}'.format(n_epoch + 1))

        for x, y in train:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            batch_loss = []
            for i in range(pred.size(0)):
                tmp_loss = criterion(pred[i], y[i])
                batch_loss.append(tmp_loss)

            loss = torch.mean(torch.stack(batch_loss))
            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())
            losses.append(loss.item())

            progress_bar.set_postfix(train_loss=np.mean(losses[-500:]))

            progress_bar.update(x.shape[0])

        progress_bar.close()

        for x, y in test:

            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                pred = model(x)

            pred_values, pred_indices = pred.max(2)
            pred_cpu = pred_indices.cpu()
            pred_cpu = filter_single_tags(pred_cpu)
            y_cpu = y.cpu()

            test_preds.extend(pred_cpu)
            test_targets.extend(y_cpu.numpy())

            batch_loss = []
            for i in range(pred.size(0)):
                tmp_loss = criterion(pred[i], y[i])
                batch_loss.append(tmp_loss)

            loss = torch.mean(torch.stack(batch_loss))
            test_losses.append(loss.item())

        mean_test_loss = np.mean(test_losses)

        test_targets = np.concatenate(test_targets).squeeze()
        test_targets[test_targets == 0] = 1
        test_preds = np.concatenate(test_preds).squeeze()

        print('\n')
        print(classification_report(test_targets, test_preds))

        print()
        print('Losses: train - {:.3f}, test - {:.3f}'.format(np.mean(train_losses), mean_test_loss))

        # Early stopping:
        if mean_test_loss < best_test_loss:
            best_test_loss = mean_test_loss
        else:
            print('Early stopping')
            break

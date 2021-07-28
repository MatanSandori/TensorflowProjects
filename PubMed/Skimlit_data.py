import sys
sys.path.append(".");

import tensorflow as tf

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

import pathlib

class data:
    def __init__(self):
        pass;
    def Load(self, shuffle=True):
        data_path = pathlib.Path("Learning-2/NPL/Datasets/NLP - Pudmud20k/PubMed_20k_RCT_numbers_replaced_with_at_sign");
        train_data_path = pathlib.Path("Learning-2/NPL/Datasets/NLP - Pudmud20k/PubMed_20k_RCT_numbers_replaced_with_at_sign/train.txt");
        test_data_path = pathlib.Path("Learning-2/NPL/Datasets/NLP - Pudmud20k/PubMed_20k_RCT_numbers_replaced_with_at_sign/test.txt");
        validation_data_path = pathlib.Path("Learning-2/NPL/Datasets/NLP - Pudmud20k/PubMed_20k_RCT_numbers_replaced_with_at_sign/dev.txt");

        def GetLines(file_path):
            with open(file_path, "r") as f:
                return f.readlines();

        train_data_f_txt = GetLines(train_data_path);
        test_data_f_text = GetLines(test_data_path);
        validation_data_f_text = GetLines(validation_data_path);

        self.class_names = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"];
        class_names = self.class_names;

        def GetData(data_f_txt):
            classes = [];
            labels = [];
            texts = [];
            for line in data_f_txt:
                for i in range(len(class_names)):
                    r = line.find(f"{class_names[i]}");
                    if(r == 0):
                        classes.append(class_names[i]);
                        labels.append(i);
                        texts.append(line.replace(f"{class_names[i]}\t", ""));

            return classes, labels, texts;

        train_classes, train_labels, train_text = GetData(train_data_f_txt);
        test_classes, test_labels, test_text = GetData(test_data_f_text);
        validation_classes, validation_labels, validation_text = GetData(validation_data_f_text);

        def SetData(classes, labels, text):
            data_df = pd.DataFrame({
                "classes": classes,
                "labels": labels,
                "text": text
            });

            if(shuffle == True):
                data_df = data_df.sample(frac=1, random_state=42);

            labels_np, sentances_np = data_df["labels"].to_numpy(), data_df["text"].to_numpy();

            labels_np = labels_np.astype(np.int);

            return data_df, sentances_np, labels_np;

        self.train_df , self.train_sentances_np, self.train_labels_np = SetData(train_classes, train_labels, train_text);
        self.test_df, self.test_sentances_np, self.test_labels_np = SetData(test_classes, test_labels, test_text);
        self.val_df , self.val_sentances_np, self.val_labels_np = SetData(validation_classes, validation_labels, validation_text);

    def GetTrain(self):
        return self.train_df , self.train_sentances_np, self.train_labels_np;

    def GetTest(self):
        return self.test_df, self.test_sentances_np, self.test_labels_np;
    
    def GetValidation(self):
        return self.val_df , self.val_sentances_np, self.val_labels_np;

    def GetClassNames(self):
        return self.class_names;

    def SplitChar(self, text):
        res = [];
        for sen in text:
            res.append(" ".join(sen));
        return res;

    def GetScore(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="micro"),
            "recall": recall_score(y_true, y_pred, average="micro"),
            "f1-score": f1_score(y_true, y_pred, average="micro")
        };
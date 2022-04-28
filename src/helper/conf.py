from sklearn import tree

TRAIN_VALUES_PATH = "./data/raw/train_values.csv"
TRAIN_LABELS_PATH = "./data/raw/train_labels.csv"
TEST_VALUES_PATH = "./data/raw/test_values.csv"
TEMPLATE_SUBMISSION_PATH = "./data/raw/submission_format.csv"

model = tree.DecisionTreeClassifier(max_depth=5,random_state=42,criterion='entropy')

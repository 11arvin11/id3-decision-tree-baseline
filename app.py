
import math
from collections import Counter

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CBC Classification System",
    page_icon="🩸",
    layout="wide",
)

FEATURES = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn",
    "RBC", "HGB", "HCT", "MCV", "MCH",
    "MCHC", "PLT", "PDW", "PCT",
]


# --------------------------------------------------
# NUMERICAL ID3 ALGORITHM
# --------------------------------------------------

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def information_gain(data, feature, target, threshold):
    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    if left.empty or right.empty:
        return 0.0

    parent_entropy = entropy(data[target])

    weighted_entropy = (
        (len(left) / len(data)) * entropy(left[target])
        + (len(right) / len(data)) * entropy(right[target])
    )

    return parent_entropy - weighted_entropy


def best_split(data, features, target):
    best_split_result = None
    best_gain = 0.0

    for feature in features:
        values = sorted(data[feature].unique())

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            gain = information_gain(
                data, feature, target, threshold
            )

            if gain > best_gain:
                best_gain = gain
                best_split_result = (feature, threshold)

    return best_split_result


def build_tree(data, features, target, depth=0, max_depth=10):
    if data.empty:
        return None

    if data[target].nunique() == 1:
        return data[target].iloc[0]

    majority_class = data[target].mode().iloc[0]

    if depth >= max_depth or not features:
        return majority_class

    split = best_split(data, features, target)

    if split is None:
        return majority_class

    feature, threshold = split

    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    if left.empty or right.empty:
        return majority_class

    return {
        "feature": feature,
        "threshold": threshold,
        "default": majority_class,
        "left": build_tree(
            left, features, target, depth + 1, max_depth
        ),
        "right": build_tree(
            right, features, target, depth + 1, max_depth
        ),
    }


def predict(tree, record):
    if tree is None:
        raise ValueError("The decision tree could not make a prediction.")

    while isinstance(tree, dict):
        feature = tree["feature"]

        if feature not in record:
            raise ValueError(f"Missing input feature: {feature}")

        if record[feature] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]

    return tree


# --------------------------------------------------
# LOAD DATASET AND TRAIN BASELINE MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    data = pd.read_csv("diagnosed_cbc_screened.csv")
    data.columns = data.columns.astype(str).str.strip()

    required_columns = FEATURES + ["Diagnosis"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    for feature in FEATURES:
        data[feature] = pd.to_numeric(
            data[feature], errors="coerce"
        )

    data = data.dropna()

    if data["Diagnosis"].nunique() < 2:
        raise ValueError(
            "The dataset needs at least two diagnosis classes."
        )

    train_data, _ = train_test_split(
        data,
        test_size=0.25,
        random_state=42,
        stratify=data["Diagnosis"],
    )

    tree = build_tree(
        train_data,
        FEATURES,
        "Diagnosis",
        max_depth=10,
    )

    return tree, len(train_data), len(data)


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title("CBC-Based Hematological Classification")
st.caption("Numerical ID3 Baseline | Academic Research Prototype")

st.warning(
    "RESEARCH PROTOTYPE ONLY: This system has not been clinically "
    "validated. Do not use its predictions to diagnose patients "
    "or guide medical treatment."
)

try:
    tree, training_count, dataset_count = load_model()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.info(
        "Check that diagnosed_cbc_screened.csv is located in "
        "the same folder as app.py."
    )
    st.stop()


with st.expander("About this system"):
    st.write(
        "This prototype uses a numerical ID3-style decision tree "
        "to classify CBC measurements into a category found in "
        "the research dataset."
    )

    st.write(f"Dataset records loaded: {dataset_count}")
    st.write(f"Records used to train the tree: {training_count}")
    st.write(f"Input features: {len(FEATURES)}")

st.divider()
st.subheader("Enter CBC Measurements")

st.write(
    "Enter the measurements from a CBC record using the "
    "same units and conventions as the dataset."
)

# Initialize inputs for the reset button.
if "cbc_reset_count" not in st.session_state:
    st.session_state.cbc_reset_count = 0

reset_key = st.session_state.cbc_reset_count

with st.form(key=f"cbc_form_{reset_key}"):

    columns = st.columns(3)
    values = {}

    for index, feature in enumerate(FEATURES):
        with columns[index % 3]:
            values[feature] = st.number_input(
                feature,
                min_value=0.0001,
                value=1.0,
                format="%.4f",
                key=f"{feature}_{reset_key}",
            )

    predict_clicked = st.form_submit_button(
        "Predict Classification",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if predict_clicked:
    st.divider()
    st.subheader("Prediction Results")

    try:
        prediction = predict(tree, values)

        st.success(
            f"Predicted dataset class: {prediction}"
        )

        st.caption(
            "This is the model's predicted dataset category, "
            "not a confirmed medical diagnosis."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# RESET INPUTS
# --------------------------------------------------

if st.button("Reset Inputs", use_container_width=True):
    st.session_state.cbc_reset_count += 1
    st.rerun()


st.divider()

st.caption(
    "Academic research use only | Numerical ID3 baseline | "
    "Clinical validity has not been established."
)

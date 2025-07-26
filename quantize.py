import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

# Step 1: Load trained scikit-learn model
sk_model = joblib.load("sklearn_model.joblib")
coef = sk_model.coef_
intercept = sk_model.intercept_

# Step 2: Save unquantized parameters
unquant_params = {
    'coef': coef,
    'intercept': intercept
}
joblib.dump(unquant_params, "unquant_params.joblib")

# Step 3: Quantize coef (manual uint8 quantization)
min_w = np.min(coef)
max_w = np.max(coef)
scale_w = 255 / (max_w - min_w)

quantized_coef = np.round((coef - min_w) * scale_w).astype(np.uint8)

# Step 4: Save quantized parameters (only coef is quantized)
quant_params = {
    'coef': quantized_coef,
    'scale_w': scale_w,
    'min_w': min_w
}
joblib.dump(quant_params, "quant_params.joblib")

# Step 5: Dequantize the weights
dequant_coef = quantized_coef.astype(np.float32) / scale_w + min_w
dequant_intercept = intercept  # use original intercept directly

# Step 6: Define a single-layer PyTorch model
class QuantLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(dequant_coef[np.newaxis, :], dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor([dequant_intercept], dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

# Step 7: Prepare data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Step 8: Run inference using quantized PyTorch model
model = QuantLinearModel()
model.eval()
inputs = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    outputs = model(inputs).squeeze().numpy()
    r2 = r2_score(y_test, outputs)
    print(f"R² Score (Quantized Model): {r2}")

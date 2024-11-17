import pandas as pd
from  sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# بارگذاری فایل txt با جداکننده کاما
data = pd.read_csv('household_power_consumption.txt', sep=';')

# نمایش 5 ردیف اول داده‌ها
#print(data.head())


#حذف ستون‌های غیرضروری
data = data.drop(columns=['Date', 'Time'])

# شناسایی ستون‌هایی که حاوی مقادیر غیر عددی هستند
def clean_non_numeric(data):
    for column in data.columns:
        # تبدیل مقادیر غیر عددی به NaN
        data[column] = pd.to_numeric(data[column], errors='coerce')
    return data

# پاکسازی داده‌ها (جایگزین کردن مقادیر غیر عددی با NaN)
data_cleaned = clean_non_numeric(data)


# جایگزینی NaN با میانگین هر ستون
data_filled = data_cleaned.fillna(data_cleaned.mean())



# انتخاب ستون‌هایی که می‌خواهیم نرمال‌سازی کنیم
columns_to_normalize = ['Global_active_power', 'Global_reactive_power', 
                        'Global_intensity', 'Sub_metering_1', 
                        'Sub_metering_2', 'Sub_metering_3']


# ایجاد یک نمونه از MinMaxScaler
scaler = MinMaxScaler()

# نرمال‌سازی داده‌ها
data_filled[columns_to_normalize] = scaler.fit_transform(data_filled[columns_to_normalize])

# نمایش داده‌های نرمال‌شده
print(data_filled)



# انتخاب ویژگی‌ها و هدف
X = data_filled[['Global_reactive_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
y = data_filled['Global_active_power']

# تقسیم داده‌ها به مجموعه‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ایجاد مدل رگرسیون خطی
model_lr = LinearRegression()

# آموزش مدل
model_lr.fit(X_train, y_train)

# پیش‌بینی
y_pred = model_lr.predict(X_test)

# ارزیابی مدل
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

from sklearn.tree import DecisionTreeRegressor

# ساخت مدل درخت تصمیم
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# پیش‌بینی و ارزیابی
y_pred_dt = model_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f'Mean Squared Error (Decision Tree): {mse_dt}')


# ساخت مدل شبکه عصبی
model_nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model_nn.fit(X_train, y_train)

# پیش‌بینی و ارزیابی
y_pred_nn = model_nn.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Mean Squared Error (Neural Network): {mse_nn}')


# پیش‌بینی‌ها با استفاده از داده‌های تست برای هر مدل
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_nn = model_nn.predict(X_test)

# محاسبه MAE، MSE و R² برای هر مدل
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

# نمایش نتایج
print("Linear Regression:")
print(f"R²: {r2_lr}")
print(f"MAE: {mae_lr}")
print(f"MSE: {mse_lr}")

print("\nDecision Tree:")
print(f"R²: {r2_dt}")
print(f"MAE: {mae_dt}")
print(f"MSE: {mse_dt}")

print("\nNeural Network:")
print(f"R²: {r2_nn}")
print(f"MAE: {mae_nn}")
print(f"MSE: {mse_nn}")


# رسم نمودار پیش‌بینی‌ها در برابر مقادیر واقعی
def plot_prediction_vs_true(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', label='Perfect Prediction (45 degree line)')
    plt.title(f'{model_name}: Prediction vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# رسم نمودار خطاها (Error Plot)
def plot_error(y_true, y_pred, model_name):
    errors = y_pred - y_true
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, errors, color='green', alpha=0.5, label='Errors')
    plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
    plt.title(f'{model_name}: Error Plot')
    plt.xlabel('True Values')
    plt.ylabel('Prediction Error (Predicted - True)')
    plt.legend()
    plt.grid(True)
    plt.show()

# رسم نمودارها برای هر مدل

# برای رگرسیون خطی
plot_prediction_vs_true(y_test, y_pred_lr, 'Linear Regression')
plot_error(y_test, y_pred_lr, 'Linear Regression')

# برای درخت تصمیم
plot_prediction_vs_true(y_test, y_pred_dt, 'Decision Tree')
plot_error(y_test, y_pred_dt, 'Decision Tree')

# برای شبکه عصبی
plot_prediction_vs_true(y_test, y_pred_nn, 'Neural Network')
plot_error(y_test, y_pred_nn, 'Neural Network')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_excel("ERA_Sample_Data_from_Customer.xlsx")
data = df[['district', 'CashTendered']].dropna()
districts = data['district'].value_counts().nlargest(3).index.tolist()
plt.figure(figsize=(10, 6))

for d in districts:
    temp = data[data['district'] == d].reset_index(drop=True)
    x = np.arange(len(temp)).reshape(-1, 1)
    y = temp['CashTendered'].values
    # Fit linear model
    model = LinearRegression().fit(x, y)
    # Forecast next 15 points
    future_x = np.arange(len(temp), len(temp) + 15).reshape(-1, 1)
    pred = model.predict(future_x)

    # Plot old and new
    plt.plot(x, y, label=f'{d} (history)')
    plt.plot(future_x, pred, '--', label=f'{d} (forecast)')

plt.title('Cash Tendered Forecast (Next 15 points)')
plt.xlabel('Transaction Order')
plt.ylabel('Cash Tendered')
plt.legend()
plt.show()

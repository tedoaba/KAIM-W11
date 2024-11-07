from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_data(data, columns, scaler_type="standard"):
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data, scaler

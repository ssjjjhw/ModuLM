import json
import numpy as np

def maeRmse(path):
    def parse_property(value):
        """
        提取 <start_property> 和 <end_property> 之间的数值。
        """
        start_idx = value.find("<start_property>") + len("<start_property>")
        end_idx = value.find("<end_property>")
        return float(value[start_idx:end_idx])

    def calculate_mae_rmse(predictions, targets):
        """
        计算 MAE 和 RMSE。
        """
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
        return mae, rmse

    # 读取数据并提取数值
    predictions = []
    targets = []

    # with open("/data/cz/moltc/results/chchpre0.01mlp.txt.txt", "r") as file:
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line.strip())
            
            # 提取 prediction 和 target 中的数值
            prediction_value = parse_property(data["prediction"])
            target_value = parse_property(data["target"])
            
            # 保存提取的数值
            predictions.append(prediction_value)
            targets.append(target_value)

    # 计算 MAE 和 RMSE
    mae, rmse = calculate_mae_rmse(predictions, targets)
    return mae,rmse
# path = "/data/cz/moltc/results/huiguitest.txt"
# mae, rmse = mseRmse(path)
# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")

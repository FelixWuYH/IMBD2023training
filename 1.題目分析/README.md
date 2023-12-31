### 1.題目分析
#### 1-1.accumulation_hour.[csv](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.題目分析/accumlation_hour.csv)
1. [欄位值統計](https://chat.openai.com/share/79b79838-97b7-40a3-8859-839ab8408540)
2. 時間序列視覺化 [不同爐的觀點](https://chat.openai.com/share/a88c0bd4-021a-4689-92f1-be943a33cf5f)
- 產線一
![產線一](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View1Line1.png)
___
- 產線二
![產線二](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View1Line2.png)
___
3. 時間序列視覺化 不同層的觀點
- 產線一
![產線一](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View2Line1.png)
___
- 產線二
![產線二](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View2Line2.png)
___
#### 1-2.prediction.[csv](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.題目分析/prediction.csv)
* 預測各爐的異常燈管總數
#### 1-3.anomaly_train.[csv](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.題目分析/anomaly_train.csv)
1. [欄位值統計]()
2. 時間序列視覺化 [不同爐的觀點](https://chat.openai.com/share/7adca566-559b-4d9d-8a6e-2423499e9b50)
- 產線一
![產線一](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View1Dot1.png)
___
- 產線二
![產線二](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View1Dot2.png)
___
2. 時間序列視覺化 [不同層的觀點](https://chat.openai.com/share/7adca566-559b-4d9d-8a6e-2423499e9b50)
- 產線一
![產線一](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View2Dot1.png)
___
- 產線二
![產線二](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.%E9%A1%8C%E7%9B%AE%E5%88%86%E6%9E%90/View2Dot2.png)
___
#### 1-4.power.[csv](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.題目分析/power.csv)
* 每個爐層的燈管功率總和可能是造成異常燈管的物理因素，可以由累積使用時數推算。
#### 1-5.cooler.[csv](https://github.com/FelixWuYH/IMBD2023training/blob/main/1.題目分析/cooler.csv)
* 每個爐層都有唯一的水冷板進水流量、入水溫度、迴水溫度及AB點水溫，須經過計算得到特徵。

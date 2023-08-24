import datetime as dt
import platform
from copy import deepcopy
import csv

#       日期,   oven_id,   layer_id,   lamp_id,     累積時數,  壞掉燈管數
#   進去水量,   進去水溫,   出來水溫,  水冷板A溫度,  水冷板B溫度
#    燈管功率
TITLE_NAME = ['日期', 'oven_id', 'layer_id', '壞掉燈管數(低功率)', '壞掉燈管數(高功率)', '進水量', '進水溫', '出水溫', '水冷板A溫度', '水冷板B溫度', '累積時數', '燈管功率(低功率)', '燈管功率(高功率)']
OVEN_ID = ['1B0', '1C0', '1D0', '1E0', '1G0', '2B0', '2C0', '2D0', '2E0', '2G0']


class File:
    def __init__(self):
        def read(fileName: str):
            with open('2022_Final_Big_Data/ProjectA txt/' + fileName) as f:
                title = f.readline().replace('\n', '').split(',')
                data: list[str] = []
                for i in f.readlines():
                    data.append(i.replace('\n', '').split(','))
            return title, data

        self.anomaly_train1_title, self.anomaly_train1_data = read('anomaly_train1.txt')
        self.cooler_title, self.cooler_data = read('cooler.txt')
        self.power_title, self.power_data = read('power.txt')
        self.accumulation_hour1_title, self.accumulation_hour1_data = read('accumulation_hour1.txt')

    def get_cooler(self, oven_id, layer_id):
        index = self.cooler_title.index(oven_id)
        layer_id = int(layer_id) - 1
        water_volume = self.cooler_data[layer_id][index]
        if layer_id < 10:
            in_temperature = self.cooler_data[20][index]
            out_temperature = self.cooler_data[21][index]
        else:
            in_temperature = self.cooler_data[22][index]
            out_temperature = self.cooler_data[23][index]
        A_temperature = self.cooler_data[25 + layer_id * 2 - 1][index]
        B_temperature = self.cooler_data[25 + layer_id * 2][index]

        return [water_volume, in_temperature, out_temperature, A_temperature, B_temperature]

    def get_power(self, accumulation_hour):
        if accumulation_hour < 0:
            return self.power_data[0][2:4]
        for line in self.power_data:
            if int(accumulation_hour) >= int(line[1].split('-')[0]) and int(accumulation_hour) <= int(line[1].split('-')[1]):
                return line[2:4]


file = File()


class Accumulation_hour:
    def __init__(self):
        self.__map = {}
        for data in file.anomaly_train1_data:
            self.__add_index(data[0], data[1], data[2], data[4])
        for data in file.accumulation_hour1_data:
            self.__add_index(data[0], data[1], data[2], data[3])

    def __add_index(self, date, oven_id, layer_id, target):
        if date not in self.__map:
            self.__map[date] = {}
        if oven_id not in self.__map[date]:
            self.__map[date][oven_id] = {}
        self.__map[date][oven_id][layer_id] = target

    def get(self, date: str, oven_id: str, layer_id: str):
        try:
            return self.__map[date][oven_id][layer_id]
        except:
            return '0'


accumulation_hour = Accumulation_hour()


class Num_of_broken_lamps:
    def __init__(self):
        self.__higher_power = {'1', '2', '60', '61', '62', '63', '121', '122'}
        self.__map = {}
        for data in file.anomaly_train1_data:
            self.__add_index(data[0], data[1], data[2], data[3])

    def __add_index(self, date: str, oven_id: str, layer_id: str, ids: str):
        if date not in self.__map:
            self.__map[date] = {}
        if oven_id not in self.__map[date]:
            self.__map[date][oven_id] = {}
        if oven_id not in self.__map[date][oven_id]:
            self.__map[date][oven_id][layer_id] = [0, 0]
        for id in ids.split('_'):
            if id in self.__higher_power:
                self.__map[date][oven_id][layer_id][1] += 1
            else:
                self.__map[date][oven_id][layer_id][0] += 1

    def get(self, date: str, oven_id: str, layer_id: str):
        try:
            return [str(i) for i in self.__map[date][oven_id][layer_id]]
        except:
            return ['0', '0']


num_of_broken_lamps = Num_of_broken_lamps()


class Preprocess:
    def single_day(self, date: str, new_data: list):
        data = ['0' for i in range(len(TITLE_NAME))]
        data[0] = date
        for oven_id in OVEN_ID:
            data[1] = oven_id
            for layer_id in range(1, 20):
                data[2] = str(layer_id)
                data[3:5] = num_of_broken_lamps.get(date, oven_id, data[2])
                data[5:10] = file.get_cooler(oven_id, layer_id)
                data[10] = accumulation_hour.get(date, oven_id, data[2])

                new_data.append(deepcopy(data))

    def initialize(self):
        start_date = file.anomaly_train1_data[0]
        end_date = file.anomaly_train1_data[-1]
        new_data = []
        dt_today = dt.datetime.strptime(start_date[0], "%Y/%m/%d").date()
        dt_next_day = dt.datetime.strptime(end_date[0], "%Y/%m/%d").date()
        if platform.system() == 'Linux':
            while dt_today <= dt_next_day:
                self.single_day(dt_today.strftime('%Y/%-m/%-d'), new_data)
                dt_today += dt.timedelta(days=1)
        else:
            while dt_today <= dt_next_day:
                self.single_day(dt_today.strftime('%Y/%#m/%#d'), new_data)
                dt_today += dt.timedelta(days=1)
        for i in range(190):
            try:
                self.fill_accumulation_hours(new_data, i)
            except:
                pass
        for data in new_data:
            data[11:13] = file.get_power(int(data[10]))
        return new_data

    def fill_accumulation_hours(self, new_data, offset: int):
        def fill_with_average(accumulation_hours: list[int], interval_start: int, interval_end: int):
            average = (accumulation_hours[interval_end] - accumulation_hours[interval_start]) / (interval_end - interval_start)
            result = []
            current_value = accumulation_hours[interval_start]
            for i in range(interval_end - interval_start):
                result.append(round(current_value) if current_value > 0 else 0)
                current_value += average
            return result

        accumulation_hours = []
        counter = 0
        while counter < len(new_data):
            accumulation_hours.append(0 if new_data[counter + offset][10] == '0' else int(new_data[counter + offset][10]))
            counter += 190
        interval_start = None
        interval_end = None
        first_nonzero_index = None
        last_nonzero_index = None
        for i in range(len(accumulation_hours)):
            if accumulation_hours[i] != 0:
                if first_nonzero_index is None:
                    first_nonzero_index = i
                last_nonzero_index = i
                # 填入閉區間
                interval_start = interval_end
                interval_end = i
                if interval_start is not None:
                    accumulation_hours[interval_start:interval_end] = fill_with_average(accumulation_hours, interval_start, interval_end)

        # 填入開區間
        if first_nonzero_index is not None and last_nonzero_index is not None and first_nonzero_index < last_nonzero_index:
            average = (accumulation_hours[last_nonzero_index] - accumulation_hours[first_nonzero_index]) / (last_nonzero_index - first_nonzero_index)
            if average < 0:
                raise ValueError
            current_value = accumulation_hours[first_nonzero_index]
            while first_nonzero_index >= 0:
                accumulation_hours[first_nonzero_index] = round(current_value) if current_value > 0 else 0
                current_value -= average
                first_nonzero_index -= 1

            current_value = accumulation_hours[last_nonzero_index]
            while last_nonzero_index < len(accumulation_hours):
                accumulation_hours[last_nonzero_index] = round(current_value)
                current_value += average
                last_nonzero_index += 1

        # 寫回原陣列
        counter = 0
        index = 0
        while counter < len(new_data):
            new_data[counter + offset][10] = str(accumulation_hours[index])
            counter += 190
            index += 1


if __name__ == "__main__":
    preprocess = Preprocess()
    all_data = preprocess.initialize()

    with open("data.csv", "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(TITLE_NAME)
        writer.writerows(all_data)

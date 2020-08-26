import glob
import os
import os.path
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import tee
import numpy as np
import matplotlib.ticker as ticker
import cv2 as cv
from sklearn.linear_model import LinearRegression
import scipy
import scipy.stats
from datetime import datetime
import time

def get_thermal_image(file_name):
    thermal_suffix = "_thermal.png"
    fn_prefix, _ = os.path.splitext(file_name)
    thermal_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + thermal_suffix)
    return thermal_filename

def get_downscaled_image(file_name):
    downscaled_suffix = "_rgb_image_downscaled.jpg"
    fn_prefix, _ = os.path.splitext(file_name)
    downscaled_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + downscaled_suffix)
    return downscaled_filename

def get_annotated_image(file_name):
    annotated_suffix = "_L.png"
    fn_prefix, _ = os.path.splitext(file_name)
    annotated_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + annotated_suffix)
    return annotated_filename

def get_thermal_csv(file_name):
    csv_suffix = '_thermal_values.csv'
    fn_prefix, _ = os.path.splitext(file_name)
    thermal_csv_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + csv_suffix)
    return thermal_csv_filename

def get_metadata_csv(file_name):
    csv_suffix = '_metadata.csv'
    fn_prefix, _ = os.path.splitext(file_name)
    metadata_csv_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + csv_suffix)
    return metadata_csv_filename   

def get_air_temp(file_name):
    metadata_filename = get_metadata_csv(file_name)
    data = pd.read_csv(metadata_filename, sep=',',header=0, parse_dates=False)
    return float(data.loc[0, 'AtmosphericTemperature'].split(' ')[0])

def get_hist_plot(data, min_data, max_data):
    x = data
    # set bin boundaries
    bin_vals = np.arange(min_data - 1, max_data + 1, 1)

    fig, ax = plt.subplots()
    y_vals, x_vals, e_ = ax.hist(data, bins=bin_vals, edgecolor='black')
    # print(y_vals)
    # print(x_vals)
    y_max = round((max(y_vals) / len(data)) + 0.02, 2)
    ax.set_xlabel("Temperature(C)")
    ax.set_ylabel("Pixel number percent")
    ax.set_yticks(ticks=np.arange(0.0, y_max * len(data), 0.025 * len(data)))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(data)))
    # plt.show()

def train(image_path_list):
    chunks = []
    df = pd.DataFrame(chunks, columns=['Temp', 'RGB'])
    for i in range(0,100):
        thermal_csv_filename = get_thermal_csv(image_path_list[i])   
        rgb_filename = get_downscaled_image(image_path_list[i])      
        data = pd.read_csv(thermal_csv_filename, sep=',',header=0, parse_dates=False) 
        annotated_image = Image.open(get_annotated_image(image_path_list[i]))
        downscaled_img = cv.resize(np.array(annotated_image), dsize=(80, 60), interpolation=cv.INTER_AREA)
        h, w, ch = downscaled_img.shape
        for i in range(h):
            for j in range(w):
                row = data.loc[((data['x'] == i) & (data['y'] == j))]
                temp = row['Temp(c)'].values[0]
                r = row['R'].values[0]
                g = row['G'].values[0]
                b = row['B'].values[0]
                if not row.empty:
                    boo = np.array(downscaled_img[i,j]) == np.array([0,255,0])                    
                    if boo.all():                        
                        new_row = {'Temp': temp, 'RGB' : r+g+b}
                        if ((df['Temp']==new_row['Temp']) & (df['RGB']==new_row['RGB'])).any():
                            pass
                        else:
                            df = df.append(new_row, ignore_index=True)   
   
    np.save('Histogram_paper/parameters/sunlit_vals', df)

if __name__ == "__main__":
    start_train = datetime.now()
    print("Training start : ", start_train)
    image_path_list = glob.glob("train/*.jpg")
    pd.set_option('display.max_rows', 10000)    

    if not(os.path.exists("Histogram_paper/parameters/sunlit_vals.npy")):
        train(image_path_list)  

    end_train = datetime.now()
    print("Training end : ", end_train)
    print('Train Duration: {}'.format(end_train - start_train))   

    start_test = datetime.now()
    print("Testing start : ", start_test)
    image_path_list = glob.glob("test/*.jpg")

    actual_temp = np.load('Histogram_paper/parameters/sunlit_vals.npy', allow_pickle=True)
    act_df = pd.DataFrame(actual_temp, columns = ['Temp', 'RGB'])
    color = act_df['RGB']
    trained_temp = np.mean(act_df['Temp'])
    # print(trained_temp)

    for image_path in image_path_list:
        thermal_image = Image.open(get_thermal_image(image_path))
        thermal_csv_filename = get_thermal_csv(image_path)
        air_temperature = get_air_temp(image_path)
        pd.set_option('display.max_rows', 10000)
        data = pd.read_csv(thermal_csv_filename, sep=',',header=0, parse_dates=False)
        # get_hist_grad_thers(data)
        temps = data["Temp(c)"].sort_values()
        min_temp = round(min(temps)) - 1
        max_temp = round(max(temps)) + 1
        bins = max_temp - min_temp

        # get_hist_plot(temps, min(temps), max(temps))
        prev_rpc = 0

        chunks = []
        df = pd.DataFrame(chunks, columns=['Temp', 'Count %', 'RPC'])

        total = len(temps)

        for i in range (min_temp, max_temp):
            count = (len([t for t in temps if i <= t < i+1]) / len(data)) * 100
            new_row = {'Temp': i, 'Count %' : count}
            df = df.append(new_row, ignore_index=True)

        for i in df.index:
            if i < 1:
                df['RPC'][i] = 0
                continue

            numerator = df['Count %'][i] - df['Count %'][i-1]
            denominator = df['Temp'][i] - df['Temp'][i-1]

            rpc = numerator/denominator
            df['RPC'][i] = prev_rpc + rpc
            prev_rpc = prev_rpc + rpc
        
        # df.sort_values('RPC', inplace=True)
        # print(df)
        # df.set_index('Temp')['RPC'].plot(figsize=(12, 10), linewidth=2.5, color='maroon')
        # plt.xlabel("Temp", labelpad=15)
        # plt.ylabel("RPC", labelpad=15)
        # plt.show()        
        
        min_rpc = min(df['RPC'])        
        max_rpc = max(df['RPC'])        

        filtered_df = pd.DataFrame(chunks, columns=['RPC', 'tw', 'td', 'tc', 'RMSE'])        

        for rpc in np.arange(min_rpc,max_rpc):
            # print('rpc = ', rpc)
            for i in range(df.index.start, df.index.stop - 1, 1):                
                i_rpc = df['RPC'][i]
                if (i == df.index.start or i == df.index.stop - 1 or abs(i_rpc - rpc) < abs(df['RPC'][i+1] - rpc) and abs(i_rpc - rpc) < abs(df['RPC'][i-1] - rpc)):
                    # print('i = ', i)                    
                    for j in range(df.index.stop - 1, df.index.start, -1):                        
                        if j <= i:
                            break
                        j_rpc = df['RPC'][j]                        
                        if (j == df.index.stop - 1 or j == df.index.start or abs(j_rpc - rpc) < abs(df['RPC'][j+1] - rpc) and abs(j_rpc - rpc) < abs(df['RPC'][j-1] - rpc)):
                            # print('j = ', j)
                            subset = df.loc[i:j, :]
                            arr = subset['Temp']
                            if arr.count() > 0:
                                tc = sum(arr)/len(arr)
                                a = pow((tc - trained_temp), 2)
                                rmse = np.sqrt(a)
                                new_row = {'RPC': rpc, 'tw' : df['Temp'][i], 'td' : df['Temp'][j], 'tc' : tc, 'RMSE' : rmse}                                
                                filtered_df = filtered_df.append(new_row, ignore_index=True)
                                
        row = filtered_df[filtered_df['RMSE'] == min(filtered_df['RMSE'])]

        # print(filtered_df)
        # print(row)  

        templ = min(row['tw'])
        # print(templ) 
        tempu = max(row['td']) 
        # print(tempu)        
        
        # get_hist_plot(temps, templ + 1, tempu - 1) 
        # plt.show()     
        # temps = temps[(temps >= templ) & (temps <= tempu)]
        data = data[(data['Temp(c)'] >= templ) & (data['Temp(c)'] <= tempu)]
        data = data[(data['Temp(c)'] >= (air_temperature - 7)) & (data['Temp(c)'] <= (air_temperature + 7))]
        # print(len(data))

        sum_column = data['R']+data['G']+data['B']
        data["RGB"] = sum_column        

        data = data[(data['RGB']).isin(color)]
        # print(min(data['Temp(c)']))
        # print(max(data['Temp(c)']))
        # print(len(data))

        downscaled_image = Image.open(get_downscaled_image(image_path))
        mask = np.zeros_like(np.asarray(downscaled_image))
        row = mask.shape[0]
        col = mask.shape[1]

        for r in range(row):
            for c in range(col):
                temp_data = data.loc[((data['x'] == r) & (data['y'] == c)),'Temp(c)'].values.tolist()
                if temp_data != []:
                    mask[r,c,:] = [0, 255, 0]
                else:
                    mask[r,c,:] = [165,42,42]

        fn_prefix, _ = os.path.splitext(image_path)
        downscaled_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + '_hist_downscaled.jpg')
        upscaled_filename = os.path.join(fn_prefix + '/' + fn_prefix.split('\\')[1] + '_hist_upscaled.jpg')

        downscaled = Image.fromarray((mask).astype(np.uint8))
        downscaled.save(downscaled_filename)
        d = Image.open(downscaled_filename) 
        # plt.imshow(d)
        plt.show()
        d_ar = np.asarray(d)
        # u = d.size(504,342)
        u= cv.resize(d_ar, dsize=(480, 320), interpolation=cv.INTER_AREA)
        im = Image.fromarray(u)
        im.save(upscaled_filename)
        
        # plt.imshow(u)
        # plt.show()
        

    end_test = datetime.now()   
    print("Testing end : ", end_test)
    print('Test Duration: {}'.format(end_test - start_test))
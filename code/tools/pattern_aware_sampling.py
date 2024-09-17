# 标记显著点文件

#找下一个可标记点j
# 返回值：final_index: 接下来的可标记点j
def find_j(i, data, time, delta): # i：上一个可选点
    index = i+1 # 从上一个可选点往后走1步
    # 当index已经是最后一个点或者i本来已经是最后一个点的时候，就把data中最后一个点纳入标记点
    if index >= len(data):
        return len(data)-1
    max_slope = max((data[index]+delta-data[i])/(time[index]-time[i]), (data[index]-delta-data[i])/(time[index]-time[i]))
    min_slope = min((data[index]+delta-data[i])/(time[index]-time[i]), (data[index]-delta-data[i])/(time[index]-time[i]))
    final_index = index
    index += 1
    while index < len(data):
        slope = (data[index]-data[i])/(time[index]-time[i])
        # 如果符合equation5的式子，即新线斜率落进了可行空间中，即上一个可选点i和当前步点index点之间所有数据点都不超过误差，即当前这个点可以选做可标记点j
        # 那么，按照论文，要选的点应该是最后一个可选点，所以继续往前推，依照equation6更新llow和lup，在这里是max_slope和min_slope
        if min_slope <= slope <= max_slope:
            final_index = index
            max_slope = min((data[index] + delta - data[i]) / (time[index] - time[i]), max_slope)
            min_slope = max(min_slope, (data[index] - delta - data[i]) / (time[index] - time[i]))
            index += 1
        # 超出了可行空间的点，那么就应该跳出查找，选择上一个可行点了
        else:
            final_index = index
            break

    return final_index

# data:
# time: timestamp, the i in the paper
# delta: error tolerance
# sample_index就是选取的点集
def sample_points(data, time, delta):
    index = 0
    sample_index = [index]
    while index < len(data)-1:
        next_point = find_j(index, data, time, delta)
        sample_index.append(next_point)
        index = next_point

    return sample_index





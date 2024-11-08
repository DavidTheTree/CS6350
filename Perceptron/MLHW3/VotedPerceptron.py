import numpy as np
import pandas as pd
from Main_HW3 import load_data


# Voted Perceptron Training Function
def voted_perceptron_train(train_data, T=2, learning_rate=1.0):
    # Initialize weights and other parameters
    weights = np.zeros(train_data.shape[1] - 1)
    bias = 0
    weight_vectors = []  # Stores distinct weight vectors and their counts
    count = 1  # Initialize the count of current weight vector

    # Convert labels to {-1, 1} for Perceptron
    train_labels = train_data['genuineorforged'].apply(lambda x: 1 if x == 1 else -1)

    for epoch in range(T):
        for index, row in train_data.iterrows():
            x = row[:-1].values
            y = train_labels.iloc[index]
            # Perceptron update rule
            if y * (np.dot(weights, x) + bias) <= 0:
                # Misclassified example: save current weights with its count
                weight_vectors.append((weights.copy(), bias, count))
                # Update weights and reset count
                weights += learning_rate * y * x
                bias += learning_rate * y
                count = 1
            else:
                # Correct classification, increment count
                count += 1

    # Append the final weight vector
    weight_vectors.append((weights.copy(), bias, count))
    return weight_vectors


# Voted Perceptron Prediction Function
def voted_perceptron_predict(test_data, weight_vectors):
    predictions = []
    for _, row in test_data.iterrows():
        x = row[:-1].values
        # Voting mechanism
        vote = sum(count * (1 if np.dot(weights, x) + bias > 0 else -1)
                   for weights, bias, count in weight_vectors)
        # Final prediction based on majority vote
        prediction = 1 if vote > 0 else -1
        predictions.append(prediction)
    return predictions


# Calculate average prediction error
def calculate_error(predictions, true_labels):
    true_labels = true_labels.apply(lambda x: 1 if x == 1 else -1)
    errors = sum(predictions != true_labels)
    return errors / len(true_labels)


# Load the data
train_data, test_data, attributes = load_data()

# Train the Voted Perceptron
weight_vectors = voted_perceptron_train(train_data)

# Make predictions on the test data
test_labels = test_data['genuineorforged']
predictions = voted_perceptron_predict(test_data, weight_vectors)

# Calculate average test error
error = calculate_error(np.array(predictions), test_labels)

# Display weight vectors and their counts
print("Distinct weight vectors and their counts:")
for i, (weights, bias, count) in enumerate(weight_vectors):
    print(f"Weight vector {i + 1}: {weights}, Bias: {bias}, Count: {count}")

print("Average test error on the test dataset:", error)
'''
Distinct weight vectors and their counts:
Weight vector 1: [0. 0. 0. 0.], Bias: 0, Count: 1
Weight vector 2: [ -3.8481 -10.1539   3.8561   4.2228], Bias: -1.0, Count: 2
Weight vector 3: [-3.800092 -8.5502   -4.6195    3.46722 ], Bias: -2.0, Count: 1
Weight vector 4: [-5.066792 -5.7319   -7.0455    1.58102 ], Bias: -1.0, Count: 8
Weight vector 5: [-7.119692 -1.8934   -7.84094   0.36722 ], Bias: 0.0, Count: 3
Weight vector 6: [-10.388892 -14.634      7.71636    0.2254  ], Bias: 1.0, Count: 7
Weight vector 7: [-14.309292 -10.5617     7.47958   -1.8897  ], Bias: 2.0, Count: 5
Weight vector 8: [-17.775992  -6.4893     3.19138   -3.4315  ], Bias: 1.0, Count: 9
Weight vector 9: [-17.658162  -4.9104    -4.83862   -3.403469], Bias: 0.0, Count: 10
Weight vector 10: [-16.208062  -1.3037    -8.89432   -5.000069], Bias: 1.0, Count: 2
Weight vector 11: [-20.884562  -6.9673     2.07468   -5.334559], Bias: 2.0, Count: 1
Weight vector 12: [-18.236662 -17.1047     3.40568    0.136141], Bias: 1.0, Count: 3
Weight vector 13: [-18.120742 -13.8828    -0.02452   -2.709559], Bias: 2.0, Count: 1
Weight vector 14: [-20.708742 -10.0174    -0.35812   -3.989259], Bias: 3.0, Count: 8
Weight vector 15: [-20.587482  -9.79393   -0.83139   -3.019019], Bias: 4.0, Count: 2
Weight vector 16: [-21.078292  -6.94873   -4.47499   -6.119419], Bias: 5.0, Count: 10
Weight vector 17: [-18.883992  -2.39843   -9.45099   -8.844819], Bias: 6.0, Count: 2
Weight vector 18: [-22.905792 -10.70243    3.10401  -10.354719], Bias: 7.0, Count: 3
Weight vector 19: [-22.795472  -8.72833   -0.26279  -11.007309], Bias: 8.0, Count: 14
Weight vector 20: [-22.494662  -8.55452   -2.01699  -10.518099], Bias: 9.0, Count: 6
Weight vector 21: [-21.894162  -6.62182   -5.30579  -10.842249], Bias: 10.0, Count: 11
Weight vector 22: [-20.838962  -5.43612   -7.94689  -10.731919], Bias: 11.0, Count: 17
Weight vector 23: [-20.890941 -12.48822   -5.89279   -7.581119], Bias: 10.0, Count: 27
Weight vector 24: [-22.034141  -8.74692  -11.47049   -6.945339], Bias: 9.0, Count: 30
Weight vector 25: [-19.386241 -18.88432  -10.13949   -1.474639], Bias: 8.0, Count: 3
Weight vector 26: [-18.419161 -15.04172  -15.07089   -5.606939], Bias: 9.0, Count: 12
Weight vector 27: [-18.748401 -10.58652  -19.64269   -4.618139], Bias: 8.0, Count: 6
Weight vector 28: [-20.458801 -15.36452  -13.43179   -4.220739], Bias: 9.0, Count: 7
Weight vector 29: [-19.899411 -15.67492  -13.24872   -3.774209], Bias: 10.0, Count: 31
Weight vector 30: [-20.418911 -12.41162  -16.33822   -2.789309], Bias: 9.0, Count: 13
Weight vector 31: [-23.561211 -25.44812   -0.66092   -3.450959], Bias: 10.0, Count: 2
Weight vector 32: [-25.286911 -20.97842   -8.88282   -1.643659], Bias: 9.0, Count: 1
Weight vector 33: [-32.329011 -11.77842   -8.62349   -6.326859], Bias: 10.0, Count: 30
Weight vector 34: [-30.649111  -7.57162  -13.16329   -8.719959], Bias: 11.0, Count: 53
Weight vector 35: [-29.297311  -6.51212  -15.50699   -8.319979], Bias: 12.0, Count: 15
Weight vector 36: [-32.803311 -19.07882   -0.34639   -9.072139], Bias: 13.0, Count: 24
Weight vector 37: [-31.070211 -15.12442   -5.08759  -11.573839], Bias: 14.0, Count: 18
Weight vector 38: [-32.891811  -8.64962  -13.13899  -11.155289], Bias: 13.0, Count: 17
Weight vector 39: [-32.104911 -18.21592   -9.35229   -3.651889], Bias: 12.0, Count: 5
Weight vector 40: [-30.597211 -16.25632  -12.41069   -3.774319], Bias: 13.0, Count: 18
Weight vector 41: [-29.585511 -15.35412  -14.76129   -3.347179], Bias: 14.0, Count: 104
Weight vector 42: [-27.193811 -10.79762  -19.75009   -6.245879], Bias: 15.0, Count: 21
Weight vector 43: [-30.799111 -16.77162   -9.65849   -7.074339], Bias: 16.0, Count: 27
Weight vector 44: [-30.638031 -10.30922  -18.01579   -5.552739], Bias: 15.0, Count: 27
Weight vector 45: [-34.334131 -23.98712   -0.43629   -8.170839], Bias: 16.0, Count: 12
Weight vector 46: [-33.439011 -19.21332   -5.27939  -13.761739], Bias: 17.0, Count: 1
Weight vector 47: [-33.958481 -15.95002   -8.36889  -12.776819], Bias: 16.0, Count: 113
Weight vector 48: [-34.287681 -11.49482  -12.94069  -11.788019], Bias: 15.0, Count: 1
Weight vector 49: [-32.256681  -9.64282  -15.95279  -11.785016], Bias: 16.0, Count: 10
Weight vector 50: [-31.893891 -17.93232  -14.03149   -8.451816], Bias: 15.0, Count: 28
Weight vector 51: [-29.876191 -16.13412  -16.98959   -8.241916], Bias: 16.0, Count: 48
Weight vector 52: [-28.656391 -14.03592  -20.18499   -8.113486], Bias: 17.0, Count: 7
Weight vector 53: [-27.869491 -23.60222  -16.39829   -0.610086], Bias: 16.0, Count: 29
Weight vector 54: [-34.205891 -14.31742  -16.384015  -7.394486], Bias: 17.0, Count: 24
Weight vector 55: [-31.557991 -24.45482  -15.053015  -1.923786], Bias: 16.0, Count: 28
Weight vector 56: [-30.107891 -20.84812  -19.108715  -3.520386], Bias: 17.0, Count: 27
Weight vector 57: [-27.913591 -16.29782  -24.084715  -6.245786], Bias: 18.0, Count: 2
Weight vector 58: [-31.935391 -24.60182  -11.529715  -7.755686], Bias: 19.0, Count: 23
Weight vector 59: [-31.334891 -22.66912  -14.818515  -8.079836], Bias: 20.0, Count: 11
Weight vector 60: [-30.279691 -21.48342  -17.459615  -7.969506], Bias: 21.0, Count: 11
Weight vector 61: [-28.790091 -18.05462  -21.490515  -9.395406], Bias: 22.0, Count: 78
Weight vector 62: [-29.119331 -13.59942  -26.062315  -8.406606], Bias: 21.0, Count: 6
Weight vector 63: [-30.829731 -18.37742  -19.851415  -8.009206], Bias: 22.0, Count: 38
Weight vector 64: [-31.349231 -15.11412  -22.940915  -7.024306], Bias: 21.0, Count: 13
Weight vector 65: [-34.491531 -28.15062   -7.263615  -7.685956], Bias: 22.0, Count: 2
Weight vector 66: [-36.217231 -23.68092  -15.485515  -5.878656], Bias: 21.0, Count: 6
Weight vector 67: [-37.146931 -19.88382  -20.128415  -5.582956], Bias: 20.0, Count: 25
Weight vector 68: [-35.467031 -15.67702  -24.668215  -7.976056], Bias: 21.0, Count: 20
Weight vector 69: [-38.697531 -22.89052  -13.024915  -8.922186], Bias: 22.0, Count: 23
Weight vector 70: [-38.052811 -18.28432  -21.371915  -6.212286], Bias: 21.0, Count: 10
Weight vector 71: [-36.701011 -17.22482  -23.715615  -5.812306], Bias: 22.0, Count: 74
Weight vector 72: [-35.914111 -26.79112  -19.928915   1.691094], Bias: 21.0, Count: 5
Weight vector 73: [-34.406411 -24.83152  -22.987315   1.568664], Bias: 22.0, Count: 58
Weight vector 74: [-33.724611 -19.98112  -28.200615  -4.535636], Bias: 23.0, Count: 26
Weight vector 75: [-36.261911 -26.94012  -19.395215  -3.006736], Bias: 24.0, Count: 38
Weight vector 76: [-33.870211 -22.38362  -24.384015  -5.905436], Bias: 25.0, Count: 53
Weight vector 77: [-34.389711 -19.12032  -27.473515  -4.920536], Bias: 24.0, Count: 22
Weight vector 78: [-38.085811 -32.79822   -9.894015  -7.538636], Bias: 25.0, Count: 12
Weight vector 79: [-37.190691 -28.02442  -14.737115 -13.129536], Bias: 26.0, Count: 1
Weight vector 80: [-37.710161 -24.76112  -17.826615 -12.144616], Bias: 25.0, Count: 113
Weight vector 81: [-38.039361 -20.30592  -22.398415 -11.155816], Bias: 24.0, Count: 1
Weight vector 82: [-36.008361 -18.45392  -25.410515 -11.152813], Bias: 25.0, Count: 38
Weight vector 83: [-33.990661 -16.65572  -28.368615 -10.942913], Bias: 26.0, Count: 49
Weight vector 84: [-34.711341 -23.41402  -22.527815 -10.319223], Bias: 27.0, Count: 114
Weight vector 85: [-32.517041 -18.86372  -27.503815 -13.044623], Bias: 28.0, Count: 2
Weight vector 86: [-36.538841 -27.16772  -14.948815 -14.554523], Bias: 29.0, Count: 34
Weight vector 87: [-35.483641 -25.98202  -17.589915 -14.444193], Bias: 30.0, Count: 11
Weight vector 88: [-33.994041 -22.55322  -21.620815 -15.870093], Bias: 31.0, Count: 63
Weight vector 89: [-31.346141 -32.69062  -20.289815 -10.399393], Bias: 30.0, Count: 15
Weight vector 90: [-31.675381 -28.23542  -24.861615  -9.410593], Bias: 29.0, Count: 44
Weight vector 91: [-32.194881 -24.97212  -27.951115  -8.425693], Bias: 28.0, Count: 49
Weight vector 92: [-32.714381 -21.70882  -31.040615  -7.440793], Bias: 27.0, Count: 17
Weight vector 93: [-35.944881 -28.92232  -19.397315  -8.386923], Bias: 28.0, Count: 23
Weight vector 94: [-35.300161 -24.31612  -27.744315  -5.677023], Bias: 27.0, Count: 211
Weight vector 95: [-32.908461 -19.75962  -32.733115  -8.575723], Bias: 28.0, Count: 11
Weight vector 96: [-36.665761 -28.05122  -22.429915  -8.195133], Bias: 29.0, Count: 37
Weight vector 97: [-36.504681 -21.58882  -30.787215  -6.673533], Bias: 28.0, Count: 27
Weight vector 98: [-40.200781 -35.26672  -13.207715  -9.291633], Bias: 29.0, Count: 12
Weight vector 99: [-39.305661 -30.49292  -18.050815 -14.882533], Bias: 30.0, Count: 1
Weight vector 100: [-39.825131 -27.22962  -21.140315 -13.897613], Bias: 29.0, Count: 113
Weight vector 101: [-40.154331 -22.77442  -25.712115 -12.908813], Bias: 28.0, Count: 1
Weight vector 102: [-38.123331 -20.92242  -28.724215 -12.90581 ], Bias: 29.0, Count: 38
Weight vector 103: [-36.105631 -19.12422  -31.682315 -12.69591 ], Bias: 30.0, Count: 49
Weight vector 104: [-36.826311 -25.88252  -25.841515 -12.07222 ], Bias: 31.0, Count: 6
Weight vector 105: [-36.039411 -35.44882  -22.054815  -4.56882 ], Bias: 30.0, Count: 29
Weight vector 106: [-42.375811 -26.16402  -22.04054  -11.35322 ], Bias: 31.0, Count: 52
Weight vector 107: [-40.925711 -22.55732  -26.09624  -12.94982 ], Bias: 32.0, Count: 3
Weight vector 108: [-38.277811 -32.69472  -24.76524   -7.47912 ], Bias: 31.0, Count: 24
Weight vector 109: [-36.083511 -28.14442  -29.74124  -10.20452 ], Bias: 32.0, Count: 125
Weight vector 110: [-36.412751 -23.68922  -34.31304   -9.21572 ], Bias: 31.0, Count: 6
Weight vector 111: [-38.123151 -28.46722  -28.10214   -8.81832 ], Bias: 32.0, Count: 38
Weight vector 112: [-38.642651 -25.20392  -31.19164   -7.83342 ], Bias: 31.0, Count: 13
Weight vector 113: [-41.784951 -38.24042  -15.51434   -8.49507 ], Bias: 32.0, Count: 2
Weight vector 114: [-43.510651 -33.77072  -23.73624   -6.68777 ], Bias: 31.0, Count: 6
Weight vector 115: [-44.440351 -29.97362  -28.37914   -6.39207 ], Bias: 30.0, Count: 25
Weight vector 116: [-42.760451 -25.76682  -32.91894   -8.78517 ], Bias: 31.0, Count: 20
Weight vector 117: [-45.990951 -32.98032  -21.27564   -9.7313  ], Bias: 32.0, Count: 23
Weight vector 118: [-45.346231 -28.37412  -29.62264   -7.0214  ], Bias: 31.0, Count: 49
Weight vector 119: [-43.613131 -24.41972  -34.36384   -9.5231  ], Bias: 32.0, Count: 35
Weight vector 120: [-42.826231 -33.98602  -30.57714   -2.0197  ], Bias: 31.0, Count: 5
Weight vector 121: [-41.318531 -32.02642  -33.63554   -2.14213 ], Bias: 32.0, Count: 122
Weight vector 122: [-38.926831 -27.46992  -38.62434   -5.04083 ], Bias: 33.0, Count: 21
Weight vector 123: [-42.532131 -33.44392  -28.53274   -5.86929 ], Bias: 34.0, Count: 27
Weight vector 124: [-42.371051 -26.98152  -36.89004   -4.34769 ], Bias: 33.0, Count: 27
Weight vector 125: [-46.067151 -40.65942  -19.31054   -6.96579 ], Bias: 34.0, Count: 12
Weight vector 126: [-45.172031 -35.88562  -24.15364  -12.55669 ], Bias: 35.0, Count: 1
Weight vector 127: [-45.691501 -32.62232  -27.24314  -11.57177 ], Bias: 34.0, Count: 113
Weight vector 128: [-46.020701 -28.16712  -31.81494  -10.58297 ], Bias: 33.0, Count: 1
Weight vector 129: [-43.989701 -26.31512  -34.82704  -10.579967], Bias: 34.0, Count: 38
Weight vector 130: [-41.972001 -24.51692  -37.78514  -10.370067], Bias: 35.0, Count: 52
Weight vector 131: [-43.811101 -33.60522  -28.54354  -10.474387], Bias: 36.0, Count: 84
Weight vector 132: [-42.361001 -29.99852  -32.59924  -12.070987], Bias: 37.0, Count: 152
Weight vector 133: [-42.690241 -25.54332  -37.17104  -11.082187], Bias: 36.0, Count: 6
Weight vector 134: [-44.400641 -30.32132  -30.96014  -10.684787], Bias: 37.0, Count: 38
Weight vector 135: [-44.920141 -27.05802  -34.04964   -9.699887], Bias: 36.0, Count: 49
Weight vector 136: [-45.439641 -23.79472  -37.13914   -8.714987], Bias: 35.0, Count: 17
Weight vector 137: [-48.670141 -31.00822  -25.49584   -9.661117], Bias: 36.0, Count: 23
Weight vector 138: [-48.025421 -26.40202  -33.84284   -6.951217], Bias: 35.0, Count: 84
Weight vector 139: [-47.238521 -35.96832  -30.05614    0.552183], Bias: 34.0, Count: 5
Weight vector 140: [-45.730821 -34.00872  -33.11454    0.429753], Bias: 35.0, Count: 82
Weight vector 141: [-44.993221 -29.15622  -37.91314   -5.236147], Bias: 36.0, Count: 38
Weight vector 142: [-49.370521 -34.67292  -26.97414   -5.644347], Bias: 37.0, Count: 2
Weight vector 143: [-46.978821 -30.11642  -31.96294   -8.543047], Bias: 38.0, Count: 53
Weight vector 144: [-47.498321 -26.85312  -35.05244   -7.558147], Bias: 37.0, Count: 22
Weight vector 145: [-51.194421 -40.53102  -17.47294  -10.176247], Bias: 38.0, Count: 12
Weight vector 146: [-50.299301 -35.75722  -22.31604  -15.767147], Bias: 39.0, Count: 1
Weight vector 147: [-50.818771 -32.49392  -25.40554  -14.782227], Bias: 38.0, Count: 113
Weight vector 148: [-51.147971 -28.03872  -29.97734  -13.793427], Bias: 37.0, Count: 1
Weight vector 149: [-49.116971 -26.18672  -32.98944  -13.790424], Bias: 38.0, Count: 38
Weight vector 150: [-47.099271 -24.38852  -35.94754  -13.580524], Bias: 39.0, Count: 55
Weight vector 151: [-46.312371 -33.95482  -32.16084   -6.077124], Bias: 38.0, Count: 81
Weight vector 152: [-44.862271 -30.34812  -36.21654   -7.673724], Bias: 39.0, Count: 61
Weight vector 153: [-48.220471 -37.58852  -24.77464   -8.244854], Bias: 40.0, Count: 13
Weight vector 154: [-46.730871 -34.15972  -28.80554   -9.670754], Bias: 41.0, Count: 78
Weight vector 155: [-47.060111 -29.70452  -33.37734   -8.681954], Bias: 40.0, Count: 44
Weight vector 156: [-47.579611 -26.44122  -36.46684   -7.697054], Bias: 39.0, Count: 13
Weight vector 157: [-50.721911 -39.47772  -20.78954   -8.358704], Bias: 40.0, Count: 8
Weight vector 158: [-51.651611 -35.68062  -25.43244   -8.063004], Bias: 39.0, Count: 25
Weight vector 159: [-49.971711 -31.47382  -29.97224  -10.456104], Bias: 40.0, Count: 3
Weight vector 160: [-50.491211 -28.21052  -33.06174   -9.471204], Bias: 39.0, Count: 124
Weight vector 161: [-49.704311 -37.77682  -29.27504   -1.967804], Bias: 38.0, Count: 5
Weight vector 162: [-48.196611 -35.81722  -32.33344   -2.090234], Bias: 39.0, Count: 82
Weight vector 163: [-47.459011 -30.96472  -37.13204   -7.756134], Bias: 40.0, Count: 40
Weight vector 164: [-45.067311 -26.40822  -42.12084  -10.654834], Bias: 41.0, Count: 11
Weight vector 165: [-48.824611 -34.69982  -31.81764  -10.274244], Bias: 42.0, Count: 37
Weight vector 166: [-48.663531 -28.23742  -40.17494   -8.752644], Bias: 41.0, Count: 27
Weight vector 167: [-52.359631 -41.91532  -22.59544  -11.370744], Bias: 42.0, Count: 12
Weight vector 168: [-51.464511 -37.14152  -27.43854  -16.961644], Bias: 43.0, Count: 1
Weight vector 169: [-51.983981 -33.87822  -30.52804  -15.976724], Bias: 42.0, Count: 113
Weight vector 170: [-52.313181 -29.42302  -35.09984  -14.987924], Bias: 41.0, Count: 1
Weight vector 171: [-50.282181 -27.57102  -38.11194  -14.984921], Bias: 42.0, Count: 93
Weight vector 172: [-49.495281 -37.13732  -34.32524   -7.481521], Bias: 41.0, Count: 81
Weight vector 173: [-48.045181 -33.53062  -38.38094   -9.078121], Bias: 42.0, Count: 27
Weight vector 174: [-45.850881 -28.98032  -43.35694  -11.803521], Bias: 43.0, Count: 2
Weight vector 175: [-49.872681 -37.28432  -30.80194  -13.313421], Bias: 44.0, Count: 45
Weight vector 176: [-48.383081 -33.85552  -34.83284  -14.739321], Bias: 45.0, Count: 78
Weight vector 177: [-48.712321 -29.40032  -39.40464  -13.750521], Bias: 44.0, Count: 44
Weight vector 178: [-49.231821 -26.13702  -42.49414  -12.765621], Bias: 43.0, Count: 13
Weight vector 179: [-52.374121 -39.17352  -26.81684  -13.427271], Bias: 44.0, Count: 8
Weight vector 180: [-53.303821 -35.37642  -31.45974  -13.131571], Bias: 43.0, Count: 25
Weight vector 181: [-51.623921 -31.16962  -35.99954  -15.524671], Bias: 44.0, Count: 3
Weight vector 182: [-52.143421 -27.90632  -39.08904  -14.539771], Bias: 43.0, Count: 17
Weight vector 183: [-55.373921 -35.11982  -27.44574  -15.485901], Bias: 44.0, Count: 23
Weight vector 184: [-54.729201 -30.51362  -35.79274  -12.776001], Bias: 43.0, Count: 84
Weight vector 185: [-53.942301 -40.07992  -32.00604   -5.272601], Bias: 42.0, Count: 5
Weight vector 186: [-52.434601 -38.12032  -35.06444   -5.395031], Bias: 43.0, Count: 122
Weight vector 187: [-50.042901 -33.56382  -40.05324   -8.293731], Bias: 44.0, Count: 53
Weight vector 188: [-50.562401 -30.30052  -43.14274   -7.308831], Bias: 43.0, Count: 22
Weight vector 189: [-54.258501 -43.97842  -25.56324   -9.926931], Bias: 44.0, Count: 12
Weight vector 190: [-53.363381 -39.20462  -30.40634  -15.517831], Bias: 45.0, Count: 1
Weight vector 191: [-53.882851 -35.94132  -33.49584  -14.532911], Bias: 44.0, Count: 113
Weight vector 192: [-54.212051 -31.48612  -38.06764  -13.544111], Bias: 43.0, Count: 1
Weight vector 193: [-52.181051 -29.63412  -41.07974  -13.541108], Bias: 44.0, Count: 93
Weight vector 194: [-51.394151 -39.20042  -37.29304   -6.037708], Bias: 43.0, Count: 81
Weight vector 195: [-49.944051 -35.59372  -41.34874   -7.634308], Bias: 44.0, Count: 27
Weight vector 196: [-47.749751 -31.04342  -46.32474  -10.359708], Bias: 45.0, Count: 2
Weight vector 197: [-51.771551 -39.34742  -33.76974  -11.869608], Bias: 46.0, Count: 45
Weight vector 198: [-50.281951 -35.91862  -37.80064  -13.295508], Bias: 47.0, Count: 78
Weight vector 199: [-50.611191 -31.46342  -42.37244  -12.306708], Bias: 46.0, Count: 44
Weight vector 200: [-51.130691 -28.20012  -45.46194  -11.321808], Bias: 45.0, Count: 13
Weight vector 201: [-54.272991 -41.23662  -29.78464  -11.983458], Bias: 46.0, Count: 8
Weight vector 202: [-55.202691 -37.43952  -34.42754  -11.687758], Bias: 45.0, Count: 25
Weight vector 203: [-53.522791 -33.23272  -38.96734  -14.080858], Bias: 46.0, Count: 3
Weight vector 204: [-54.042291 -29.96942  -42.05684  -13.095958], Bias: 45.0, Count: 17
Weight vector 205: [-57.272791 -37.18292  -30.41354  -14.042088], Bias: 46.0, Count: 23
Weight vector 206: [-56.628071 -32.57672  -38.76054  -11.332188], Bias: 45.0, Count: 84
Weight vector 207: [-55.841171 -42.14302  -34.97384   -3.828788], Bias: 44.0, Count: 5
Weight vector 208: [-54.333471 -40.18342  -38.03224   -3.951218], Bias: 45.0, Count: 122
Weight vector 209: [-51.941771 -35.62692  -43.02104   -6.849918], Bias: 46.0, Count: 32
Weight vector 210: [-55.028371 -42.26312  -32.48054   -7.741738], Bias: 47.0, Count: 16
Weight vector 211: [-54.867291 -35.80072  -40.83784   -6.220138], Bias: 46.0, Count: 5
Weight vector 212: [-55.386791 -32.53742  -43.92734   -5.235238], Bias: 45.0, Count: 22
Weight vector 213: [-59.082891 -46.21532  -26.34784   -7.853338], Bias: 46.0, Count: 12
Weight vector 214: [-58.187771 -41.44152  -31.19094  -13.444238], Bias: 47.0, Count: 1
Weight vector 215: [-58.707241 -38.17822  -34.28044  -12.459318], Bias: 46.0, Count: 113
Weight vector 216: [-59.036441 -33.72302  -38.85224  -11.470518], Bias: 45.0, Count: 1
Weight vector 217: [-57.005441 -31.87102  -41.86434  -11.467515], Bias: 46.0, Count: 38
Weight vector 218: [-54.987741 -30.07282  -44.82244  -11.257615], Bias: 47.0, Count: 55
Weight vector 219: [-54.200841 -39.63912  -41.03574   -3.754215], Bias: 46.0, Count: 81
Weight vector 220: [-52.750741 -36.03242  -45.09144   -5.350815], Bias: 47.0, Count: 34
Weight vector 221: [-56.571041 -49.08752  -28.13314   -7.656015], Bias: 48.0, Count: 1
Weight vector 222: [-62.051841 -40.90562  -27.85496  -12.688315], Bias: 49.0, Count: 39
Weight vector 223: [-60.562241 -37.47682  -31.88586  -14.114215], Bias: 50.0, Count: 78
Weight vector 224: [-60.891481 -33.02162  -36.45766  -13.125415], Bias: 49.0, Count: 44
Weight vector 225: [-61.410981 -29.75832  -39.54716  -12.140515], Bias: 48.0, Count: 49
Weight vector 226: [-61.930481 -26.49502  -42.63666  -11.155615], Bias: 47.0, Count: 17
Weight vector 227: [-65.160981 -33.70852  -30.99336  -12.101745], Bias: 48.0, Count: 23
Weight vector 228: [-64.516261 -29.10232  -39.34036   -9.391845], Bias: 47.0, Count: 84
Weight vector 229: [-63.729361 -38.66862  -35.55366   -1.888445], Bias: 46.0, Count: 5
Weight vector 230: [-62.221661 -36.70902  -38.61206   -2.010875], Bias: 47.0, Count: 122
Weight vector 231: [-59.829961 -32.15252  -43.60086   -4.909575], Bias: 48.0, Count: 32
Weight vector 232: [-62.916561 -38.78872  -33.06036   -5.801395], Bias: 49.0, Count: 16
Weight vector 233: [-62.755481 -32.32632  -41.41766   -4.279795], Bias: 48.0, Count: 154
Weight vector 234: [-60.724481 -30.47432  -44.42976   -4.276792], Bias: 49.0, Count: 15
Weight vector 235: [-64.741781 -38.78662  -31.97506   -5.714292], Bias: 50.0, Count: 23
Weight vector 236: [-62.724081 -36.98842  -34.93316   -5.504392], Bias: 51.0, Count: 136
Weight vector 237: [-61.273981 -33.38172  -38.98886   -7.100992], Bias: 52.0, Count: 27
Weight vector 238: [-59.079681 -28.83142  -43.96486   -9.826392], Bias: 53.0, Count: 2
Weight vector 239: [-63.101481 -37.13542  -31.40986  -11.336292], Bias: 54.0, Count: 45
Weight vector 240: [-61.611881 -33.70662  -35.44076  -12.762192], Bias: 55.0, Count: 78
Weight vector 241: [-61.941121 -29.25142  -40.01256  -11.773392], Bias: 54.0, Count: 44
Weight vector 242: [-62.460621 -25.98812  -43.10206  -10.788492], Bias: 53.0, Count: 13
Weight vector 243: [-65.602921 -39.02462  -27.42476  -11.450142], Bias: 54.0, Count: 8
Weight vector 244: [-66.532621 -35.22752  -32.06766  -11.154442], Bias: 53.0, Count: 25
Weight vector 245: [-64.852721 -31.02072  -36.60746  -13.547542], Bias: 54.0, Count: 3
Weight vector 246: [-65.372221 -27.75742  -39.69696  -12.562642], Bias: 53.0, Count: 124
Weight vector 247: [-64.585321 -37.32372  -35.91026   -5.059242], Bias: 52.0, Count: 5
Weight vector 248: [-63.077621 -35.36412  -38.96866   -5.181672], Bias: 53.0, Count: 122
Weight vector 249: [-60.685921 -30.80762  -43.95746   -8.080372], Bias: 54.0, Count: 32
Weight vector 250: [-63.772521 -37.44382  -33.41696   -8.972192], Bias: 55.0, Count: 16
Weight vector 251: [-63.611441 -30.98142  -41.77426   -7.450592], Bias: 54.0, Count: 5
Weight vector 252: [-64.130941 -27.71812  -44.86376   -6.465692], Bias: 53.0, Count: 22
Weight vector 253: [-67.827041 -41.39602  -27.28426   -9.083792], Bias: 54.0, Count: 12
Weight vector 254: [-66.931921 -36.62222  -32.12736  -14.674692], Bias: 55.0, Count: 1
Weight vector 255: [-67.451391 -33.35892  -35.21686  -13.689772], Bias: 54.0, Count: 31
Weight vector 256: [-65.592991 -41.24492  -33.55256  -11.851372], Bias: 53.0, Count: 82
Weight vector 257: [-65.922191 -36.78972  -38.12436  -10.862572], Bias: 52.0, Count: 1
Weight vector 258: [-63.891191 -34.93772  -41.13646  -10.859569], Bias: 53.0, Count: 38
Weight vector 259: [-61.873491 -33.13952  -44.09456  -10.649669], Bias: 54.0, Count: 55
Weight vector 260: [-61.086591 -42.70582  -40.30786   -3.146269], Bias: 53.0, Count: 36
Average test error on the test dataset: 0.014
'''

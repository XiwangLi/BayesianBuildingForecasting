from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
################################################################################
# IM Class
################################################################################
class InverseModel:
    """
    The inverse model class that finds the change point
    """
    from scipy import optimize
    import matplotlib.pyplot as plt
    import numpy as np
    import heapq

    """
    Constructor for InverseModel
    Args:
        eui: a np array of eui consumption
        temperature: a np array of eui consumption
    """
    def __init__(self, temperature, eui):

        # Check if two arrays have the same length
        if(self.np.size(eui) != self.np.size(temperature)):
            print("Please make sure eui and temperature arrays have the same length")
        else:
            self.temperature = temperature
            self.eui = eui

    """
    This function fits a 3p (two segmented) piecewise linear regression model.
    """
    def piecewise_linear_3P(self, x, x0, y0, k1, k2):
        conds = [x < x0, x >= x0]
        funcs = [lambda x:k1*x + y0-k1*x0,
                 lambda x:k2*x + y0-k2*x0]
        return self.np.piecewise(x, conds, funcs)

    # This function fit a 4P change-point model
    def piecewise_linear_4P(self, x, x0, y0, k1, k2):
        # Breaks
        conds = [x < x0, x >= x0]
        # Functions between each breaks
        funcs = [lambda x:k1*x + y0 - k1*x0,
                 lambda x:k2*x + y0 - k2*x0]
        # Return the piecewise model
        return np.piecewise(x, conds, funcs)

    """
    This function fits a 5p (three segmented) piecewise linear regression model.
    """
    def piecewise_linear_5P(self, x, x1, x2, y0, k1, k2):
        # Breaks
        conds = [x < x1, (x >= x1) & (x <= x2), x > x2]
        # Functions between each breaks
        funcs = [lambda x:k1*x + y0 - k1*x1,
                 lambda x:y0,
                 lambda x:k2*x + y0 - k2*x2]
        # Return the piecewise model
        return np.piecewise(x, conds, funcs)

    # This function claculate the R2 for a 4P model
    def R_Squared(self, p, piecewiseF, numP):

        # print(p)

        x = self.temperature
        y = self.eui

        if(numP == 3):
            residuals = y - piecewiseF(x, p[0], p[1], p[2], p[3])
        elif(numP == 4):
            residuals = y - piecewiseF(x, p[0], p[1], p[2], p[3])
        elif(numP == 5):
            residuals = y - piecewiseF(x, p[0], p[1], p[2], p[3], p[4])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        rsquared = 1-(ss_res/ss_tot)
        return(rsquared)

    def fitIM(self, minRsquared, safe_factor = 3):
        # Initial R-squared and bestModel
        R_squared_current = 0
        bestModel = ""
        best_p = None
        best_e = None

        T_left = min(self.temperature)
        T_right = max(self.temperature)

        if(len(self.temperature) <= 5):
            safe_factor = 1

        T_left_safe = self.heapq.nsmallest(safe_factor, self.temperature)[safe_factor - 1]
        T_right_safe = self.heapq.nlargest(safe_factor, self.temperature)[safe_factor - 1]

        # print(self.heapq.nsmallest(safe_factor, self.temperature))
        # print(self.heapq.nlargest(safe_factor, self.temperature))

        # print("Safe left: " + str(T_left_safe))
        # print("Safe right: " + str(T_right_safe))

        # Loop through 3~5 P, find the best model
        for numP in range(3,6):

            # Get a least R-squared 3P model
            if(numP == 3):
                print("Fitting a 3P heating model...")
                current_model = self.piecewise_linear_3P
                # Try to fit a 3p heating profile
                current_p, current_e = optimize.curve_fit(
                        current_model, self.temperature, self.eui,
                        bounds= ([T_left_safe,  0,          -self.np.inf, -0.00001],
                                 [T_right_safe, self.np.inf, 0,            0.00001])
                        )
                # Calculate R-squared
                R_squared_new = self.R_Squared(current_p, current_model, numP)

                # If there is only less than two points outside of the
                # change point boundary, consider the points are outliers
                # 

                if(current_p[0] <= T_left_safe or
                    current_p[0] >= T_right_safe):
                    R_squared_new = -1

                print("3P heating R2 = " + str(R_squared_new))
                # Set best model to current model if R2_new > R2_current
                if(R_squared_new > R_squared_current):
                    R_squared_current = R_squared_new
                    bestModel = "3P Heating"
                    best_p = current_p
                    best_e = current_e

                print("Fitting a 3P cooling model...")
                # Try to fit a 3p cooling profile
                current_p, current_e = optimize.curve_fit(
                    current_model, self.temperature, self.eui,
                    bounds= ([T_left_safe,  0,           -0.00001,  -self.np.inf],
                             [T_right_safe, self.np.inf, 0.00001,  self.np.inf])
                    )
                # Calculate R-squared
                R_squared_new = self.R_Squared(current_p, current_model, numP)

                # If there is only less than two points outside of the
                # change point boundary, consider the points are outliers
                if(current_p[0] <= T_left_safe or
                    current_p[0] >= T_right_safe):
                    R_squared_new = -1

                print("3P cooling R2 = " + str(R_squared_new))
                # Set best model to current model if R2_new > R2_current
                if(R_squared_new > R_squared_current):
                    R_squared_current = R_squared_new
                    bestModel = "3P Cooling"
                    best_p = current_p
                    best_e = current_e

            # Get a least R-squared 4P model
            elif(numP == 4):
                current_model = self.piecewise_linear_4P
                current_p, current_e = optimize.curve_fit(
                    current_model, self.temperature, self.eui,
                    bounds = ([T_left_safe, 0,            -self.np.inf,   0.005],
                              [T_right_safe, self.np.inf, -0.005,         self.np.inf])
                    )
                # Calculate R-squared
                R_squared_new = self.R_Squared(current_p, current_model, numP)

                # If there is only less than two points outside of the
                # change point boundary, consider the points are outliers
                if(current_p[0] <= T_left_safe or
                    current_p[0] >= T_right_safe):
                    R_squared_new = -1

                # Set best model to current model if R2_new > R2_current
                if(R_squared_new > R_squared_current):
                    R_squared_current = R_squared_new
                    bestModel = "4P"
                    best_p = current_p
                    best_e = current_e

            # Get a least R-squared 4P model
            elif(numP == 5):
                print("Fitting a 5P model...")
                current_model = self.piecewise_linear_5P
                current_p, current_e = optimize.curve_fit(
                    current_model, self.temperature, self.eui,
                    bounds= ([T_left_safe,  T_left_safe,  0,      -np.inf, 0.005],
                             [T_right_safe, T_right_safe, np.inf, -0.005, np.inf])
                    )
                # Calculate R-squared
                R_squared_new = self.R_Squared(current_p, current_model, numP)

                # If there is only less than two points outside of the
                # change point boundary, consider the points are outliers
                if(current_p[0] <= T_left_safe or
                    current_p[1] >= T_right_safe):
                    R_squared_new = -1

                print("5P R2 = " + str(R_squared_new))
                # Set best model to current model if R2_new > R2_current
                if(R_squared_new > R_squared_current):
                    R_squared_current = R_squared_new
                    bestModel = "5P"
                    best_p = current_p
                    best_e = current_e
            # print("Best R2 = " + str(R_squared_current))
        # Check if the best model has an R-aquared greater than the threshold.
        if(R_squared_current >= minRsquared):
            self.cp = best_p
            self.bestModel = bestModel
            self.best_p = best_p
            self.best_e = best_e
            self.R_Squared = R_squared_current

            if(str.startswith(bestModel,"3P") or
                str.startswith(bestModel,"4P")):
                self.cp = (best_p[0], best_p[1])
                self.cpTxt = ('(' + str(round(self.cp[0], 1)) + ', ' +
                                        str(round(self.cp[1],2)) + ')')
            elif(str.startswith(bestModel, "5P")):
                self.cp = (best_p[0], best_p[2], best_p[1], best_p[2])
                self.cpTxt = ('(' + str(round(self.cp[0], 1)) + ', ' +
                                        str(round(self.cp[1],2)) + ')',
                              '(' + str(round(self.cp[2], 1)) + ', ' +
                                        str(round(self.cp[3],2)) + ')')

    """
    This function plot the inverse model and the change point
    Args:
        width: width of the plot in inches
        height: height of the plot in inches
    """
    def plotIM(self, width = 7.2, height = 4.2, save = False):
        import os
        # print(max(self.temperature))
        # Set plot size
        self.plt.figure(figsize=(width, height))
        self.axes = self.plt.gca()
        # Add labels
        self.title = 'Change-Point Model: ' + self.bestModel
        self.title += " (R-squared = " + str(round(self.R_Squared, 2)) + ")"
        self.plt.title(self.title)
        if(max(self.temperature) >= 50):
            self.plt.xlabel('Outdoor Air Temperature [F]')
        else:
            self.plt.xlabel('Outdoor Air Temperature [C]')
        self.plt.ylabel('Energy Use Intensity [kWh/(m2*day)]')
        # Add scatters and lines
        self.plt.plot(self.temperature, self.eui, "o")


        # Plot predicted line
        self.xd = np.linspace(min(self.temperature) - 1,
            max(self.temperature) + 1, 100)
        if(str.startswith(self.bestModel, "3P")):
            self.yd = self.piecewise_linear_3P(self.xd, *self.best_p)
        elif(str.startswith(self.bestModel, "4P")):
            self.yd = self.piecewise_linear_4P(self.xd, *self.best_p)
        elif(str.startswith(self.bestModel, "5P")):
            self.yd = self.piecewise_linear_5P(self.xd, *self.best_p)
        self.plt.plot(self.xd, self.yd, '-')

        # Plot change-point(s)
        if(str.startswith(self.bestModel,"3P") or
            str.startswith(self.bestModel,"4P")):
            self.plt.scatter(self.cp[0], self.cp[1],
                s=100, facecolors = 'r', edgecolors = 'none')
            self.plt.annotate(self.cpTxt, (self.cp[0], self.cp[1]))
        elif(str.startswith(self.bestModel, "5P")):
            self.plt.scatter(self.cp[0], self.cp[1],
                s = 100, facecolors = 'r', edgecolors = 'none')
            self.plt.scatter(self.cp[2], self.cp[3],
                s = 100, facecolors = 'r', edgecolors = 'none')
            self.plt.annotate(self.cpTxt[0], (self.cp[0], self.cp[1]))
            self.plt.annotate(self.cpTxt[1], (self.cp[2], self.cp[3]))
        # self.axes.set_xlim(-5, 35) # x axis starts from 0C
        self.axes.set_ylim(0, self.np.amax(self.eui) * 1.2) # Y axis starts from 0
        if(save):
            s_path = os.path.dirname(os.path.realpath(__file__))
            self.plt.savefig(s_path + '/CP.png', dpi = 100)
            self.plt.close()

    """
    This function print the inverse model's parameters.
    """
    def printIM(self):
        print("The inverse model is a", self.bestModel, "model.")
        print("The change point is:", self.cpTxt)

    # Setters
    def seteui(self, neweui):
        self.eui = neweui

    def setTemperature(self, newTemperature):
        self.temperature = newTemperature

    # Getters
    def geteui(self):
        return self.eui

    def getTemperature(self):
        return self.temperature

    def getCoeff(self):
        coeff = []
        if(self.bestModel == "3P Cooling"):
            coeff.append(self.best_p[1])    # base load
            coeff.append(self.best_p[3])    # Cooling slope
            coeff.append(self.best_p[0])    # Cooling change point
            coeff.append("NaN")             # Heating slope point
            coeff.append("NaN")             # Heating change point
        elif(self.bestModel == "3P Heating"):
            coeff.append(self.best_p[1])    # base load
            coeff.append("NaN")             # Cooling slope
            coeff.append("NaN")             # Cooling change point
            coeff.append(self.best_p[2])    # Heating slope point
            coeff.append(self.best_p[0])    # Heating change point
        elif(self.bestModel == "4P"):
            coeff.append(self.best_p[1])    # base load
            coeff.append(self.best_p[3])    # Cooling slope
            coeff.append(self.best_p[0])    # Cooling change point
            coeff.append(self.best_p[2])    # Heating slope point
            coeff.append(self.best_p[0])    # Heating change point
        elif(self.bestModel == "5P"):
            coeff.append(self.best_p[2])    # base load
            coeff.append(self.best_p[4])    # Cooling slope
            coeff.append(self.best_p[1])    # Cooling change point
            coeff.append(self.best_p[3])    # Heating slope point
            coeff.append(self.best_p[0])    # Heating change point
        return(coeff)


################################################################################
# Main process
################################################################################
# x = np.array([30.39, 30.80, 49.57, 62.63, 71.06, 76.86, 82.88, 78.21, 
#     68.81, 56.15, 42.32, 29.61], dtype=float)
# y1 = np.array([0.26738, 0.23081, 0.23657, 0.29524, 0.31960, 0.41152, 
#     0.45031, 0.41897, 0.37066, 0.25746, 0.26879, 0.28668], dtype=float)

# y2 = np.array([0.70755, 0.70632, 0.50211, 0.30893, 0.24721, 0.27732, 
#     0.25986, 0.25706, 0.26783, 0.26812, 0.47445, 0.73894], dtype=float)


# T1 = np.array([33.6108704, 34.556522, 37.20434, 44.391308, 48.93044, 
#     51.01088, 56.87384, 60.84572, 62.5478, 68.22176, 74.65226, 
#     75.40862, 76.9217, 78.2456], dtype=float)

# E1 = np.array([0.0803828, 0.120957, 0.153876, 0.0666029, 0.0819139, 
#     0.111005, 0.0581818, 0.0734928, 0.0842105, 0.0865072, 0.068134, 
#     0.0643062, 0.0872727, 0.0742584], dtype=float)

# T2 = np.array([33.006083, 35.189348, 37.372622, 44.91482, 45.311792, 
#     49.479854, 51.46466, 55.8311, 56.42654, 57.81596, 61.3886, 63.57182, 
#     68.93078, 69.52622, 71.31254, 71.31254, 71.31254, 75.08372, 76.2746, 
#     77.0684, 78.45782, 79.45016, 79.6487, 80.64104], dtype=float)

# E2 = np.array([0.301508, 0.218593, 0.221608, 0.0829146, 0.0904523, 
#     0.117588, 0.138693, 0.019598, 0.0241206, 0.0120603, 0.0105528, 
#     0.0241206, 0.0738693, 0.0497487, 0.0211055, 0.0361809, 0.0120603, 
#     0.0723618, 0.081407, 0.146231, 0.20804, 0.152261, 0.0979899, 0.129648], dtype=float)

# T3 = np.array([34.546172, 40.466012, 41.102546, 43.776032, 44.53988, 45.685652, 
#     47.467976, 48.5501, 50.52344, 56.37956, 56.57054, 56.63426, 57.07976, 59.0531, 
#     60.13508, 61.21724, 62.42666, 62.5541], dtype=float)

# E3 = np.array([1.57198, 1.74526, 1.57716, 1.68578, 1.73103, 1.71552, 1.64828, 
#     1.67284, 1.7556, 1.74397, 1.78534, 1.80474, 1.77112, 1.82672, 1.86293, 1.95345, 
#     1.8681, 1.95086], dtype=float)


# a = InverseModel(T1, E1)
# a.fitIM(0)
# a.plotIM()








# 'make zone dataset.py' - 2022 03 25
# by Dominique Massiot - CEMHTI-CNRS UPR3079, Orléans France
#
# used in the manuscript "Stabilisation of the trigonal langasite structure in Ca3Ga2-2xZnxGe4+xO14 (0 ≤ x ≤ 1) with partial ordering of three isoelectronic cations characterised by a multi-technique approach"
# by Haytem Bazzaoui, Cécile Genevois, Dominique Massiot, Vincent Sarou-Kanian, Emmanuel Veron, Sébastien Chenu, Přemysl Beran, Michael J. Pitcher and Mathieu Allix
# submitted to ??
# preprint available at ??
# scripts available in GitHub : https://github.com/DoMassiot/Fit2DLangasiteData/
#
# takes the raw data and builds the zone_data datasets used by the fitting routine
# this script has to be executed piror to [Fit2D data.py]

import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def dir_test_exists(d):
    if not os.path.exists(d):
        print(f"\tdirectory {d} did not exist - it has been created")
        os.mkdir(d)
    return d

# definition of local directories for retrieving or storing datasets
raw_data_directory = "raw_data" + os.sep  # directory where raw data can be found
data_directory = "zone_data" + os.sep  # intermediate storage for Zone data

print(f"{os.name} on {sys.platform}")
print(f"raw data taken from '{raw_data_directory}'")
print(f"Zone data will be saved as csv files in {dir_test_exists(data_directory)}")

class Zone: # the class implementing the generation of zone datasets
    _dxy = [[0, 0], [10, 42]]
    dx = 0
    dy = 0
    zone_id = -1
    zone_definition = []
    xrange = range(1, 128)
    yrange = range(0, 128)
    frame = []

    def scale(self):
        return np.array(self.xrange), np.array(self.yrange)

    def __init__(self, zoneid): # initialization of Zone self parameters
        self.zone_id = zoneid
        self.dx = self._dxy[self.zone_id][0]
        self.dy = self._dxy[self.zone_id][1]

        if self.zone_id >= 0:
            self.frame = [[40 - self.dx, 104 - self.dx], [50 - self.dy, 112 - self.dy]]
        else:
            self.frame = [[1, 128], [0, 128]]
        self.xrange = range(self.frame[0][0], self.frame[0][1])  # 40-self.dx,104-self.dx)
        self.yrange = range(self.frame[1][0], self.frame[1][1])  # 50-self.dy,112-self.dy)

        self.zone_definition = [
            [[self.frame[0][0], 53 - self.dx], [97 - self.dy, self.frame[1][1] - 1], True],
            [[self.frame[0][0], 55 - self.dx], [70 - self.dy, self.frame[1][0]], False],
            [[80 - self.dx, self.frame[0][1] - 1], [self.frame[1][0], 74 - self.dy], False],
            [[90 - self.dx, self.frame[0][1] - 1], [self.frame[1][1] - 1, 94 - self.dy], True]
        ]

    def plot_zone(self): # plots the current zone on the figure
        def exchange (xy, i, j):
            tt = xy[i]
            xy[i] = xy[j]
            xy[j] = tt
            return xy

        # rearrange table to draw the zone as a polygon
        X = [x for l in self.zone_definition for x in l[0]]
        Y = [y for l in self.zone_definition for y in l[1]]
        xy = [[X[i],Y[i]] for i in range(len(X))]
        xy = exchange(xy, 0, 1)
        xy = exchange(xy, 6, 7)
        xy.append (xy[0])
        X,Y = zip(*xy)
        plt.plot(X,Y, "brown") #"red" if self.zone_id==0 else "green")

    def select_zone(self, t, x, y): # selects the current
        def droite(x, y, pts):
            slope = (pts[1][0] - pts[1][1]) / (pts[0][0] - pts[0][1])
            if y > pts[1][0] + (x - pts[0][0]) * slope:
                return pts[2]
            return not pts[2]

        if self.zone_id >= 0:
            for z in self.zone_definition:
                if droite(x, y, z):
                    return 0
        return t[y][x]

    def plot_contour(self, table, color="black", ncontours=1):
        ly = len(table)
        lx = len(table[0])
        min_val = min([v for sublist in table for v in sublist])
        max_val = max([v for sublist in table for v in sublist])
        print(f"table is {lx}x{ly}, min: {min_val}, max:{max_val}")
        xlocal, ylocal = self.scale()
        currplot = plt.contour(xlocal, ylocal, table, ncontours, colors=color)
        return currplot

    def read_csv(self, fname):
        def split_row(row):
            return [int(i) for i in row[0].split(",")]

        with open(fname, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            t = [split_row(row) for row in spamreader]
            # remove first row and columns
            t = [[self.select_zone(t, x, y) for x in self.xrange] for y in self.yrange]
            ly = len(t)
            lx = len(t[0])
            min_val = min([v for sublist in t for v in sublist])
            max_val = max([v for sublist in t for v in sublist])
            print(f"table '{fname}' is {lx}x{ly}, min: {min_val}, max:{max_val}")
        return t

    def make_range_name(self):
        def rangetxt(r):
            return f"{r[0]},{r[-1] + 1}"

        return f"x({rangetxt(self.xrange)})-y({rangetxt(self.yrange)})"

    def save_csv(self, table):
        def make_file_name():
            return f"{data_directory}{self.nucleus} {self.make_range_name()}.csv"

        fname = make_file_name()
        with open(fname, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.xrange)
            writer.writerow(self.yrange)
            writer.writerows(table)
            print(f"table saved in '{fname}'")
        return

    def show_csv(self, nuc_name, color="black"):
        self.nucleus = nuc_name

        def make_name():
            return f"{raw_data_directory}View005 {self.nucleus} K.csv"

        table = self.read_csv(make_name())
        self.save_csv(table)
        plot = self.plot_contour(table, color)
        return table, plot

    def build(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.show_csv("Ca", color="red")
        self.show_csv("Ge", color="green")
        self.show_csv("Ga", color="blue")
        self.show_csv("Zn", color="black")
        ax.set_xlabel(f"points")
        ax.set_ylabel(f"points")
        plt.title(f"Ca-Ge-Ga-Zn {self.make_range_name()}")
        if self.zone_id >= 0:
            plt.savefig(f"{data_directory}zone{self.zone_id}-{self.make_range_name()}.jpg")
            plt.savefig(f"{data_directory}zone{self.zone_id}-{self.make_range_name()}.pdf")
        else:
            for z in [0, 1]:
                Zone(z).plot_zone()
            plt.savefig(f"{data_directory}zones.jpg")
            plt.savefig(f"{data_directory}zones.pdf")
        plt.show()


print("============================")
[Zone(z).build() for z in [-1, 0, 1]]

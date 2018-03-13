#!/usr/bin/env python
# File to construct an adjacency matrix for one specific timestep of one specific simulation trajectory.  

import pdb
import numpy as np
import matplotlib.pyplot as plt
import PIL 
import argparse
import os

def round_up_to_even(f):
    return np.ceil(f / 2.) * 2
def round_down_to_even(f):
    return np.floor(f / 2.) * 2

class Molecule:
  def __init__(self):
      self.x = []
      self.atom_types = []
      self.natoms = -1
      self.timestep = -1
      self.boxl = np.zeros((6,))
      
  def read_dump_timestep(self,fnme,targettimestep,outputfile):
    nlines = sum(1 for line in open(fnme))
    f = open(fnme)
    natoms = -1
    xlo = xhi = ylo = yhi = zlo = zhi = -1
    found_flag = False
    
    for i in range(nlines):
        line = f.readline()
        if "NUMBER OF ATOMS" in line:
            line=f.readline()
            l = line.split()
            n = int(l[0])
            self.natoms = n
            self.x = np.zeros((n,3))
            self.atom_types = np.zeros((n,))
        elif "TIMESTEP" in line:
            l = f.readline().split()
            self.timestep = int(l[0])

        elif "ITEM: BOX BOUNDS" in line:
            l = f.readline().split()
            xlo = float(l[0])
            xhi = float(l[1])
            l = f.readline().split()
            ylo = float(l[0])
            yhi = float(l[1])
            l = f.readline().split()
            zlo = float(l[0])
            zhi = float(l[1])
            self.boxl = np.array([xlo,xhi,ylo,yhi,zlo,zhi])

        elif "ITEM: ATOMS" in line:
            if self.natoms == 0:
                print "Error! no natoms!"
                exit(1)
            for i in range(self.natoms):
                line=f.readline()
                l = line.split()
                idx = int(l[0])-1
                self.atom_types[idx] = int(l[1])
                self.x[idx][0] = float(l[2])
                self.x[idx][1] = float(l[3])
                self.x[idx][2] = float(l[4])

            if targettimestep == self.timestep:
                out = open(outputfile, 'w')
                for i in range(self.natoms):
                    out.write(str(self.atom_types[i]) + '   ' + str(self.x[i][0]) + '   ' + str(self.x[i][1]))
                    out.write('\n')

                found_flag = True
                break

    f.close()

    if found_flag == False:
        print "Error! Timestep = %d not found in %s" % (targettimestep,fnme)
        exit(1)


def main():
  # main
  parser = argparse.ArgumentParser()
  parser.add_argument('trajfile',type=str, help='Trajectory file')
  parser.add_argument('timestep',type=int, help='Timestep to plot from trajectory file')
  parser.add_argument('outputfile',type=str, help='name of new output file to generate')
  args = parser.parse_args()


  mol = Molecule()

  mol.read_dump_timestep(args.trajfile,args.timestep, args.outputfile)
  #mol.plot_config(args.imagefile,trimfrac=0.1)
  # mol.plot_config(args.imagefile,trimfrac=0.40)

if __name__ == "__main__":
    main()

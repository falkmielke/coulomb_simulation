#!/usr/bin/env python3

# libraries
import os as OS
import numpy as NP
import matplotlib as MP
import matplotlib.pyplot as PLT
from tqdm import tqdm as TQDM

# global constants
dpi = 300 # image resolution
scale = 16 # plotting scatters
cm = 1/2.54 # for plotting
k_e = 1.25 * 8.987551e1 # e9 # https://en.wikipedia.org/wiki/Coulomb's_law
CoulombForce = lambda q1, q2, r: -1*q1*q2*(k_e)/(r**2) # the coulomb force
Euclid = lambda p2, p1 = NP.zeros((2,)): NP.sqrt(NP.sum(NP.power(p2-p1, 2))) # distance of two points, or vecor norm
Normed = lambda vec: vec / Euclid(vec) # unit length vector

time_step = 1e-4 # time steps
bounds = 2 # playground boundaries
Deceleration = lambda f: f * 0.98
# MinDistance = lambda *args: 0
MinDistance = lambda q1, q2: (q1.s+q2.s)/k_e
force_scaling = 0.06

count_charges = 0
collisions = 0


################################################################################
# Definitions
################################################################################

# Generic Charge object
class Charge(object):
    # this defines all attributes and capabilities of a charge.

    description = 'generische lading'
    m = 0 # mass [kg]
    q = 0 # charge [C]
    marker = '.'
    velocity = 0 # velocity [px/s]

    def __init__(self, position = (0, 0), stationary = False, wrap_x = None):
        # the constructor: gather everything that makes a charge
        self.position = NP.array(position)
        self.stationary = stationary # for simulation purposes, we can fix a charge in her place
        self.wrap_x = wrap_x
        self.ResetForces() # forces towards all other charges will be added up.


    def __str__(self):
        # a text description of the object
        return (f"{self.description} op positie {self.position}, beweging: {self.velocity}.")


    def Distance(self, from_charge):
        # the distance to another charge
        return (Euclid(self.position, from_charge.position))


    def Force(self, by_charge, append = True):
        # the force experienced due to another charge in proximity
        this_force = Normed(by_charge.position - self.position) \
                   * CoulombForce(self.q, by_charge.q, self.Distance(by_charge))

        # print (str(self), this_force)

        # either collect forces, or calculate/return
        if append:
            self.forces.append(this_force)
        else:
            return this_force


    def ResetForces(self):
        # remove all collected forces
        self.forces = [NP.zeros((2,))]


    def Move(self, extra_bounds = None):
        # all steps involced in moving this charge

        if self.stationary:
            # trivial case: the charge is stationary.
            self.ResetForces()
            return

        # get the sum of all forces
        total_force = NP.sum(NP.stack(self.forces, axis = 1), axis = 1)

        # calculate acceleration
        acceleration = total_force / self.m

        # velocity  v = a*t, with random component
        self.velocity += acceleration * time_step
        self.velocity = Deceleration(self.velocity)
        # self.velocity *= NP.random.normal(0.9, 0.2) # partial decelleration
        self.velocity += NP.random.normal(0.0, 0.1, 2) # noise

        # get the movement step
        step = self.velocity * time_step

        # calculate potential new position
        new_position = self.position + step

        global count_charges
        if self.wrap_x is not None:
            if new_position[0] > self.wrap_x[1]:
                new_position[0] -= self.wrap_x[1]-self.wrap_x[0]
                count_charges += 1
            if new_position[0] < self.wrap_x[0]:
                new_position[0] += self.wrap_x[1]-self.wrap_x[0]
                count_charges -= 1

        # collision on field boundaries
        if new_position[0] < 0:
            # x, left boundary
            new_position[0] *= -1
            self.velocity[0] *= -1
        if new_position[1] < 0:
            # y, left boundary
            new_position[1] *= -1
            self.velocity[1] *= -1
        if new_position[0] > bounds:
            # x, right boundary
            new_position[0] = bounds - (new_position[0] - bounds)
            self.velocity[0] *= -1
        if new_position[1] > bounds:
            # y, right boundary
            new_position[1] = bounds - (new_position[1] - bounds)
            self.velocity[1] *= -1

        # collision on extra boundaries
        if extra_bounds is not None:
            if new_position[0] < extra_bounds[0]:
                # x, left boundary
                new_position[0] = extra_bounds[0] + (extra_bounds[0] - new_position[0])
                self.velocity[0] *= -1
            if new_position[1] < extra_bounds[0]:
                # y, left boundary
                new_position[1] = extra_bounds[0] + (extra_bounds[0] - new_position[1])
                self.velocity[1] *= -1
            if new_position[0] > extra_bounds[1]:
                # x, right boundary
                new_position[0] = extra_bounds[1] - (new_position[0] - extra_bounds[1])
                self.velocity[0] *= -1
            if new_position[1] > extra_bounds[1]:
                # y, right boundary
                new_position[1] = extra_bounds[1] - (new_position[1] - extra_bounds[1])
                self.velocity[1] *= -1

        # new_position[0] = new_position[0] % bounds
        # new_position[1] = new_position[1] % bounds

        self.position = new_position # set the corrected position
        self.ResetForces() # empty force collection


    def Plot(self, ax, **scatter_kwargs):
        # plot the charge symbol on the graph

        ax.scatter(self.position[0], self.position[1], s=self.s*scale, marker = 'o', facecolor = 'w', zorder = 10 \
                   , **{k:v for k, v in scatter_kwargs.items() if k not in ['marker', 'facecolor', 'zorder']})
        ax.scatter(self.position[0], self.position[1], s=self.s*scale, marker = self.marker, zorder = 20 \
                   , **{k:v for k, v in scatter_kwargs.items() if k not in ['edgecolor', 'marker']})



class Electron(Charge):
    # electrons are specific types of charges.
    description = 'elektron'
    m = 1 # * 9.1093837015e−31 kg
    q = -1 # negative unit charge
    marker = '_' # the plot marker
    s = 2.818 # fm # https://en.wikipedia.org/wiki/Classical_electron_radius
    stationary = False



class Proton(Charge):
    # protons are also specific types of charges.
    description = 'proton'
    m = 1836.152 # https://en.wikipedia.org/wiki/Proton-to-electron_mass_ratio
    q = +1
    marker = '+' # the plot marker
    s = 0.877 # fm # https://en.wikipedia.org/wiki/Proton_radius_puzzle
    stationary = False



class ChargeCollection(list):
    # Here's what we can do with a set of many charges.
    extra_bounds = None
    step = 0


    def MoveAll(self ):
        # we can move all charges, based on their force interactions.

        global collisions

        for q1 in self:
            for i, q2 in enumerate(self):
                if q2 == q1:
                    continue

                ## ERROR: Major Simplification
                #  To avoide collapse, force is neglected below a certain distance.
                if q1.Distance(q2) >= MinDistance(q1, q2):
                    # only charges above a minimum distance contribute to movement.
                    q1.Force(by_charge = q2)
                else:
                    ## elastic collision
                    if q1.q != q2.q:
                        collisions += 1
                    q1.ResetForces()
                    f = q1.Force(by_charge = q2, append = False)
                    f *= NP.random.normal(0.0, 0.5, 2)
                    q1.forces.append(-f)
                    break


            # move the charge
            # note: this is double calculation, but code efficiency was not critical.
            q1.Move(self.extra_bounds)
        self.step += 1


    def PlotExtraBounds(self, ax):
        # plot the extra bounds of a material
        if self.extra_bounds is None:
            return
        x, y = self.extra_bounds
        frame = 0.02
        ax.plot([x-frame, y+frame], [x-frame, x-frame], 'k-', lw = 1)
        ax.plot([x-frame, y+frame], [y+frame, y+frame], 'k-', lw = 1)
        ax.plot([x-frame, x-frame], [x-frame, y+frame], 'k-', lw = 1)
        ax.plot([y+frame, y+frame], [x-frame, y+frame], 'k-', lw = 1)

    
    def PlotAll(self, ax, counter = False):
        # we can plot positions of the charges.
        global collisions
        global count_charges

        self.PlotExtraBounds(ax)
        for q1 in self:
            q1.Plot(ax, edgecolor = 'k', facecolor = 'k', lw = 0.5)
            for q2 in self:
                if q1 == q2:
                    continue
                if q1.stationary and q2.stationary:
                    continue
                # on non-stationary charges, plot a representation of the force vector
                a = q1.Force(q2, append = False) * time_step / q1.m
                # a = Normed(a)
                # print (a)
                a = Normed(a) * NP.log(1.+Euclid(a))
                a *= force_scaling # length adjustment for plotting
                pos = q1.position
                ax.plot([pos[0], pos[0]+a[0]], [pos[1], pos[1]+a[1]], color = '0.7', lw = 0.5, ls = '-', alpha = 0.6)

        if counter:
            ax.text(1.0, 1.95, f"ladingsteller: {count_charges: 3.0f}, tijd: {(self.step * time_step):.3f}s", ha = 'center', va = 'top') # , botsingen: {collisions: 3.0f}
        else:
            ax.text(1.0, 1.95, f"tijd: {(self.step * time_step):.3f}s", ha = 'center', va = 'top') # , botsingen: {collisions: 3.0f}


    def Animation(self, steps = 100, label = None):
        # this is an animation for a given number of time steps
        for t in TQDM(range(steps)):
            self.MoveAll()

            # if (self.step % 10) != 0:
            #     continue

            # plot
            fig = PLT.figure(figsize = (1080/dpi, 1080/dpi), dpi = dpi)
            fig.subplots_adjust(left = 0.01, bottom = 0.01, right = 0.99, top = 0.99)
            ax = fig.add_subplot(1, 1, 1, aspect = 'equal')
            # ax.spines[:].set_visible(False)

            self.PlotAll(ax, counter = label in ['conductor', 'field'])

            ax.set_xlim([0, bounds])
            ax.set_ylim([0, bounds])
            ax.set_xticks([])
            ax.set_yticks([])

            # take a photo
            folder = 'frames' if label is None else f'frames_{label}'
            fig.savefig(f'{folder}/{t:05.0f}.png', dpi = dpi)
            PLT.close()


################################################################################
# Simulations
################################################################################
def ManyCharges():
    # Simulation: many charges

    charges = []
    for x in NP.linspace(0.2, 1.2, 6, endpoint = True):
        for y in NP.linspace(0.3, 1.7, 8, endpoint = True):
            p = Proton((x, y), stationary = True)
            charges.append(p)

    for x in NP.linspace(0.6, 1.0, 3, endpoint = True):
        for y in NP.linspace(0.6, 1.4, 5, endpoint = True):
            e = Electron((x+0.01, y+0.01), stationary = False)
            charges.append(e)

    for x in NP.linspace(1.3, 1.5, 5, endpoint = True):
        for y in NP.linspace(0.9, 1.1, 5, endpoint = True):
            e = Electron((x, y), stationary = True)
            charges.append(e)
    # e1 = Electron((0.83, 0.7))
    # e2 = Electron((0.78, 1.0))
    # e3 = Electron((0.72, 1.3))
    collection = ChargeCollection(charges)# + [e1, e2, e3])

    print (collection)
    collection.Animation(steps = 2**12)


def TwoSame():
    if True:
        e1 = Electron((0.95, 1.0))
        e2 = Electron((1.05, 1.0))
    else:
        e1 = Proton((0.98, 1.0))
        e2 = Proton((1.02, 1.0))

    collection = ChargeCollection([e1, e2])
    collection.Animation(steps = 2**10)


def TwoDifferent():
    p1 = Proton((1.00, 0.95))
    p2 = Proton((0.95, 1.05))
    p3 = Proton((1.05, 1.05))
    # p5 = Proton((0.95, 0.95))
    # p6 = Proton((0.95, 1.05))
    # p7 = Proton((1.05, 1.05))
    # p8 = Proton((1.05, 0.95))
    # p9 = Proton((1.0, 1.0))
    e2 = Electron((1.25, 0.95))
    e3 = Electron((0.75, 1.05))
    e4 = Electron((0.95, 0.75))
    # p1 = Proton((1.0, 1.1))
    #e1.velocity = NP.array((100., 0.))
    #e2.velocity = NP.array((-100., 0.))

    collection = ChargeCollection([p1, p2, e2, e3, p3, e4])#, p5, p6, p7, p8, p9])
    collection.Animation(steps = 2**12)


def ChargesInConductor():
    # Simulation: many charges

    charges = []
    for x in NP.linspace(0.3, 1.2, 4, endpoint = True):
        for y in NP.linspace(0.3, 1.2, 4, endpoint = True):
            p = Proton((x, y), stationary = True)
            charges.append(p)

    for x in NP.linspace(0.3, 1.2, 4, endpoint = True):
        for y in NP.linspace(0.3, 1.2, 4, endpoint = True):
            e = Electron(NP.array([x, y])+NP.random.normal(0., 0.02, 2), stationary = False)
            charges.append(e)

    if True:
        # for x in NP.linspace(1.30, 1.50, 5, endpoint = True):
            #for y in NP.linspace(1.30, 1.50, 5, endpoint = True):
        for y in NP.linspace(0.35, 1.15, 17, endpoint = True):
            e = Electron((1.28, y), stationary = True)
            charges.append(e)

    collection = ChargeCollection(charges)# + [e1, e2, e3])
    collection.extra_bounds = [0.25, 1.25]
    collection.Animation(steps = 2**12)

"""

time_step = 1e-4 # time steps
bounds = 2 # playground boundaries
Deceleration = lambda f: f * 0.98
# MinDistance = lambda *args: 0
MinDistance = lambda q1, q2: (q1.s+q2.s)/k_e
force_scaling = 0.1


"""


def Conduction(label = None, settings = {'field': False, 'grid': False}):
    # Simulation: many charges

    charges = []
    wrap = None
    if settings['field']:
        for y in NP.linspace(0.6, 1.4, 16, endpoint = True):
            p = Electron((0.4, y), stationary = True)
            charges.append(p)
        for y in NP.linspace(0.6, 1.4, 16, endpoint = True):
            p = Proton((1.6, y), stationary = True)
            charges.append(p)
        wrap = [0.45, 1.55]

    if settings['grid']:
        for x in NP.linspace(0.5, 1.5, 6):
            for y in NP.linspace(0.5, 1.5, 6):
                p = Proton((x, y), stationary = True)
                charges.append(p)
    for _ in range(36):
        e = Electron(NP.random.uniform(0.50, 1.50, 2), stationary = False, wrap_x = wrap)
        charges.append(e)

    collection = ChargeCollection(charges)# + [e1, e2, e3])
    collection.extra_bounds = [0.45, 1.55]
    if label is not None:
        try:
            OS.system(f'rm -rf frames_{label}')
        except _:
            pass
        OS.system(f'mkdir frames_{label}')
    collection.Animation(steps = 2**13, label = label)
    if label is not None:
        OS.system(f'ffmpeg -y -r 24 -pattern_type glob -i "frames_{label}/*.png" -c:v libvpx-vp9 -crf 24 {label}.webm')




################################################################################
# Mission Control
################################################################################
if __name__ == "__main__":
    # TwoSame()
    # # ffmpeg -y -r 24 -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 -crf 24 01_electrons.webm
    # # ffmpeg -y -r 24 -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 -crf 24 02_protons.webm

    # TwoDifferent()
    # # ffmpeg -y -r 24 -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 -crf 24 03_different.webm

    # ChargesInConductor()
    # # ffmpeg -y -r 24 -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 -crf 24 04_conductor1.webm
    # # ffmpeg -y -r 24 -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 -crf 24 05_conductor2.webm

    # Conduction(label = 'conductor', settings = {'field': True, 'grid': True})
    # Conduction(label = 'field', settings = {'field': True, 'grid': False})
    Conduction(label = 'grid', settings = {'field': False, 'grid': True})
    # Conduction(label = 'electrons', settings = {'field': False, 'grid': False})
    # # ffmpeg -y -r 24 -pattern_type glob -i "frames/*.png" -c:v libvpx-vp9 -crf 24 06_conduction.webm

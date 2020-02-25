#!/usr/bin/python
#Author: Maximilian Grove
#Matrikelnummer: 380199
#WWU Münster
#Runge-Kutta Algorythmen

import numpy as np
import matplotlib.pyplot as plt

class Solver_Explicit:
    def __init__(self, function):
        self.function = function

    def Solve(self, t0, x0, T, h):
        #Auslesen der Butchertabelle
        table = self._butcher_tableau
        k_max = table.shape[1] - 1
        factors_x = np.asarray(table[:k_max, 1:])
        factors_t = table[:k_max, 0]
        factors_xnew = table[k_max, 1:]

        #Initialisierung
        n = len(x0)
        N = int((T-t0)/h)
        x = np.zeros((N+1,n))
        xadd = np.zeros(n)

        #Anfangsbedingungen
        t = np.linspace(t0,T,N+1)
        x[0,:] = x0
        for m in range(N):
            k = np.zeros((k_max,n))
            a = np.zeros(n)
            for i in range(k_max):

                #Koeffizientenberechnung
                for j in range(n):
                    a[j] = x[m,j] + h * np.dot(factors_x[i,:i-1], k[:i-1,j])
                    b = t[m] + h * factors_t[i]
                k[i] = self.function(b, a)

            #Lösung für t+h
            xadd = np.dot(k.T, factors_xnew)
            x[m+1] = x[m] + h * xadd
        return t, x

## Tableau's were taken from odespy's implementation of the Runge-Kutta version. 
## link: https://github.com/hplgit/odespy/blob/master/odespy/RungeKutta.py
class RungeKutta1(Solver_Explicit):
    quick_description = "Explizite Runge-Kutta Methode 1. Ordnung"
    _butcher_tableau = np.array(\
        [[0., 0.],
         [0., 1.]])

class RungeKutta2(Solver_Explicit):
    quick_description = "Explizite Runge-Kutta Methode 2. Ordnung"
    _butcher_tableau = np.array(\
        [[0., 0., 0.],
         [1./2., 1./2., 0.],
         [0., 0., 1.]])

class RungeKutta4(Solver_Explicit):
    quick_description = "Explizite Runge-Kutta Methode 4. Ordnung"
    _butcher_tableau = np.array(\
        [[0., 0., 0., 0., 0.],
         [1./2., 1./2., 0., 0., 0.],
         [1./2., 0., 1./2., 0., 0.],
         [1., 0., 0., 1., 0.],
         [0., 1./6., 1./3., 1./3., 1./6.]])

class Solver_Adaptive:
    def __init__(self, function):
        self.function = function

    def Solve(self, t0, x0, T, h_start, epsilon):
        #Auslesen der Butchertabelle
        table = self._butcher_tableau
        k_max = table.shape[1] - 1
        factors_x = np.asarray(table[:k_max, 1:])
        factors_t = table[:k_max, 0]
        factors_xnew1 = table[k_max, 1:]
        factors_xnew2 = table[k_max+1, 1:]

        #Initialisierung
        n = len(x0)
        solution1 = np.empty(n)
        solution2 = np.empty(n)
        x = np.empty([1,n])

        #Anfangsbedingungen
        h = h_start
        epscalc = 0
        t = np.zeros(1) + t0
        x[0,:] = x0[:]

        m = 0
        while t[m] < T:
            k1 = np.zeros((k_max,n))
            a1 = np.zeros(n)

            #Lösung mit h
            xadd1 = np.zeros(n)
            for i in range(k_max):

                for j in range(n):
                    a1[j] = x[m,j] + h * np.dot(factors_x[i,:i-1], k1[:i-1,j])
                    b1 = t[m] + h * factors_t[i]

                k1[i] = self.function(b1, a1)

            xadd1 = np.dot(k1.T, factors_xnew1)
            solution1 = x[m] + h * xadd1

            #Lösung mit h/2
            xadd2 = np.zeros(n)
            v = 0
            while v < 2:
                k2 = np.zeros((k_max,n))
                a2 = np.zeros(n)

                for i in range(k_max):

                    for j in range(n):
                        a2[j] = x[m,j] + h/2 * np.dot(factors_x[i,:i-1], k2[:i-1,j])
                        b2 = t[m] + h/2 * factors_t[i]

                    k2[i] = self.function(b2, a2)

                xadd2 += np.dot(k2.T, factors_xnew2)
                v += 1

            solution2 = x[m] + h/2 * xadd2

            #Adaptive Zeitschrittsteuerung
            epscalc = np.amax(abs(solution1 - solution2))
            if epscalc <= epsilon:
                xnew = solution2[:] + epscalc
                x = np.vstack((x, xnew))
                m += 1
                t = np.vstack((t, t[m-1] + h))
                h = 0.8 * h * (epsilon / epscalc)**(1/(1+self._order[0]))

            else:
                h = 0.8 * h * (epsilon / epscalc)**(1/(self._order[0]))
        return t, x

class Fehlberg(Solver_Adaptive):
    quick_description = "Adaptive Runge-Kutta-Fehlberg (4,5)-Methode"
    _butcher_tableau = np.array(\
        [[0., 0., 0., 0., 0., 0., 0.],
         [.25, .25, 0., 0., 0., 0., 0.],
         [.375, .09375, .28125, 0., 0., 0., 0.],
         [.92307692, .87938097, -3.27719618, 3.32089213, 0., 0., 0.],
         [1., 2.03240741,-8., 7.17348928,-.20589669, 0., 0.],
         [.5, -.2962963, 2., -1.38167641, .45297271, -.275, 0.],
         [0., .11574074, 0., .54892788, .53533138, -.2, 0.],
         [0., .11851852, 0., .51898635, .50613149, -.18, .03636364]])
    _order = (4,5)

class CashKarp(Solver_Adaptive):
    quick_description = "Adaptive Cash-Karp Runge-Kutta (4,5)-Methode"
    _butcher_tableau = np.array(
        [[0., 0., 0., 0., 0., 0., 0.],
         [.2, .2, 0., 0., 0., 0., 0.],
         [.3, .075, .225, 0., 0., 0., 0.],
         [.6, .3, -.9, 1.2, 0., 0., 0.],
         [1., -.2037037, 2.5, -2.59259259, 1.2962963, 0., 0.],
         [.875, .0294958, .34179688, .04159433, .40034541, .06176758, 0.],
         [0., .0978836, 0., .40257649, .21043771, 0., .2891022],
         [0., .10217737, 0., .3839079, .24459274, .01932199, .25]])
    _order = (4,5)

class BogackiShampine(Solver_Adaptive):
    quick_description = "Adaptive Bogacki-Shampine Runge-Kutta (2,3)-Methode"
    _butcher_tableau = np.array(
        [[0., 0., 0., 0., 0.],
         [.5, .5, 0., 0., 0.],
         [.75, 0., .75, 0., 0.],
         [1., .22222222, .33333333, .44444444, 0.],
         [0., .22222222, .33333333, .44444444, 0.],
         [0., .29166667, .25, .33333333, .125]])
    _order = (2,3)

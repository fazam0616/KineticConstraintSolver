# Kinetic Constraint Solver

This project is a C program that began as an optimized implementation of a Python test program. The original goal was to explore whether a chaotic system—such as a double pendulum—could be approximated using a constraint-based solver.

## Overview

The system models physical behavior using **nodes** and **constraints**, with support for boundaries defined by groups of nodes.

* **Nodes** are the fundamental units of the system. Each node has:

  * Surface friction
  * Radius
  * Mass

* Groups of nodes can form **walls** or **edges**, which act as boundaries that other nodes cannot cross. These enable the construction of more complex environments and interactions.

* **Constraints** define relationships between nodes. These include:

  * **Relative constraints** — maintain a fixed distance between two nodes (e.g., 10 units apart)
  * **Absolute constraints** — fix a node at a specific position
  * **Spring constraints** — apply forces based on stiffness and the distance between nodes

## Solver Approach

All node and constraint data is organized into vectors and matrices. A linear solver is then used to approximate the physical forces required to satisfy the system constraints.

By computing and distributing these forces correctly, the system can model complex physical and mechanical behavior with notable accuracy.

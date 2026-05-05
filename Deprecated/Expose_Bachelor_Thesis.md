# Exposé for Bachelor Thesis

**Title:** Development of a Predictive Safety Filter for VR Robot Teleoperation using Control Barrier Functions in Unreal Engine 5

**Student:** [Your Name]  
**Degree Program:** [Your Program, e.g., Computer Science / Robotics]  
**Date:** December 08, 2025

---

## 1. Introduction & Motivation
Teleoperation of robotic manipulators is a critical technology for hazardous environments (e.g., nuclear decommissioning, space exploration) and data collection for imitation learning. While Virtual Reality (VR) interfaces offer intuitive control, they suffer from two fundamental issues: **network latency** and **lack of physical feedback**. A user operating a robot remotely often faces a "time travel" problem: due to video transmission delay (50ms–500ms), the visual feedback lags behind the robot's actual state. Consequently, an operator may inadvertently command a collision before realizing the danger.

Traditional safety mechanisms, such as Artificial Potential Fields (APF), often result in oscillatory behavior or trap the robot in local minima. Recently, **Control Barrier Functions (CBFs)** have emerged as the state-of-the-art (SOTA) solution, providing mathematical guarantees for safety. However, integrating optimization-based CBFs directly into high-fidelity Game Engines like Unreal Engine 5—which are increasingly used as "Digital Twins" for robotics—remains an active area of research.

## 2. Problem Statement
The central problem this thesis addresses is the **"Unsafe Action Gap"** in VR teleoperation caused by latency and human error. Specifically:
1.  Direct mapping of VR hand poses to a robot end-effector is dangerous because human movements are faster/jerkier than robot constraints allow.
2.  Visual feedback delays prevent operators from reacting to collisions in real-time.
3.  Existing simulators often lack the "Predictive Safety" layer required to filter out dangerous commands before they reach the (simulated) hardware.

## 3. Objectives
The primary objective is to develop a **Predictive Safety Filter** module within Unreal Engine 5 that intercepts VR commands, modifies them using a Control Barrier Function (CBF), and ensures collision-free operation of a simulated robot arm.

**Specific Goals:**
*   **System Architecture:** Implement a "Master-Slave" Digital Twin architecture in UE5 with two robot instances:
    *   *The Ghost Robot (Master):* Low-latency, unconstrained, represents user intent.
    *   *The Shadow Robot (Slave):* High-latency, physics-constrained, represents the physical machine.
*   **Safety Algorithm:** Develop a CBF-based filter (using C++ or Python integration) that minimally alters the user’s input to satisfy safety constraints \( h(x) \geq 0 \).
*   **Validation:** Conduct a user study (n=5-10) comparing "Direct Control" vs. "Filtered Control" in a simulated pick-and-place task, measuring collision rates and task completion times.

## 4. Methodology
The project will be implemented entirely within **Unreal Engine 5**, leveraging its Chaos Physics engine for accurate collision queries.

### 4.1. Simulation Environment ("Hardware-in-the-Loop" without Hardware)
Since no physical robot is available, the thesis will utilize a rigorous **Simulation-as-Hardware** approach:
*   **Input:** VR Motion Controller (OpenXR).
*   **Robot Model:** Franka Emika Panda (7-DOF) imported via URDF.
*   **Simulated Defects:** The "Slave" robot will be artificially degraded with:
    *   *Latency Buffer:* 200ms–400ms delay on all inputs.
    *   *Motor Limits:* Capped velocity/acceleration to mimic real actuators.

### 4.2. Algorithm Implementation (The Safety Filter)
The safety filter will be formulated as a Quadratic Program (QP):
\[
u^* = \arg\min_{u} \| u - u_{human} \|^2
\]
\[
\text{s.t. } \frac{\partial h}{\partial x} f(x) + \frac{\partial h}{\partial x} g(x) u \geq -\gamma h(x)
\]
Where \( h(x) \) is the distance to the nearest obstacle. This optimization will run in real-time (60Hz+), acting as a "shield" for the Ghost Robot.

### 4.3. Evaluation Metrics
*   **Safety:** Number of collisions with static/dynamic obstacles.
*   **Efficiency:** Time to complete the task (seconds).
*   **Smoothness:** Jerk (derivative of acceleration) of the end-effector trajectory.
*   **User Load:** NASA-TLX questionnaire (optional) to assess operator frustration.

## 5. State of the Art (Related Work)
The thesis builds upon recent advancements in 2024–2025:
*   **Predictive Displays:** "Toward a Predictive eXtended Reality Teleoperation System..." (2024) [1] validates the separation of "Ghost" and "Real" views.
*   **Safety Algorithms:** "MPC-CBF with Adaptive Safety Margins" (2024) [2] establishes CBF as the SOTA for latency mitigation.
*   **Engine Utility:** "Unreal Robotics Lab" (2025) [3] confirms UE5 as a valid scientific tool for Sim-to-Real validation.

## 6. Preliminary Timeline
*   **Month 1:** Literature Review & UE5 Scene Setup (Robot import, VR Input).
*   **Month 2:** Implementation of the "Ghost vs. Shadow" architecture and Latency injection.
*   **Month 3:** Development of the Control Barrier Function (Math & Code).
*   **Month 4:** User Study (Data Collection) & Analysis.
*   **Month 5:** Writing the Thesis.

## 7. Expected Contribution
This thesis will demonstrate that **Game Engines can serve as robust safety supervisors** for robotics. By implementing a predictive CBF filter, the system is expected to eliminate collisions completely—even in the presence of significant simulated latency—proving the viability of this architecture for real-world Sim-to-Real transfer.

---
**References:**
[1] Li, et al. "Toward a Predictive eXtended Reality Teleoperation System..." (2024).
[2] Zhang, et al. "MPC-CBF with Adaptive Safety Margins..." (2024).
[3] Unreal Robotics Lab Team. "Unreal Robotics Lab: A High-Fidelity Robotics Simulator..." (2025).

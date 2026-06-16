# Simulation & Editor Controls

The simulation features a rich set of keyboard controls to navigate the 3D environment and interact with the traffic systems. 

## Camera Navigation
| Key(s) | Action |
| --- | --- |
| **W, A, S, D** | Pan the camera (Forward, Left, Backward, Right) |
| **Up, Down, Left, Right** | Pan the camera (Alternative to WASD) |
| **Q, E** | Rotate the camera horizontally (Left / Right) |
| **R, F** | Rotate the camera vertically (Pitch Up / Down) |
| **Z, X** | Zoom In (`Z`) / Zoom Out (`X`) |
| **C** | Reset the camera to the default top-down view |

## Map Editor Modes
Pressing these number keys switches your mouse click behavior. 
**Note on Adding/Deleting**: All modes act as a **smart toggle**. If you click an empty valid spot, it **Adds** the object. If you click exactly on an existing object, it **Deletes** it.

| Hotkey | Mode Name | Description |
| --- | --- | --- |
| **1** | `VIEW` (Default) | Standard view mode. Clicking on cars displays their route. Clicking on empty space allows route-planning between two points. |
| **2** | `ROADS` | Click a road segment to flip its one-way direction or toggle its drivability. |
| **3** | `ROUNDABOUTS` | Click an intersection to convert it into a roundabout. Click an existing roundabout to remove it. |
| **4** | `LIGHTS` (Traffic) | Click an intersection to place a **Traffic Light** (governs car flow). Click an existing traffic light to delete it. |
| **5** | `STOPS` | Click an intersection to place a **Stop Sign**. Click an existing stop sign to delete it. |
| **6** | `STREETLIGHTS` | Click along a road to place a physical **City Streetlight** (casts shadows and light). Click an existing streetlight to delete it. |

## Developer / Debug Toggles
| Key | Action |
| --- | --- |
| **P** | Print debug information for the nearest car to the terminal |
| **ESC** | Close the simulation |

> [!TIP]
> **Time & Lighting**
> Use the **Hour** slider on the left side of the screen to change the time of day. When transitioning from day to night, the environment skybox will update, and City Streetlights will automatically turn on their spotlights!

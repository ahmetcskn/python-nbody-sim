import matplotlib.pyplot as plt

plt.style.use("dark_background")
import numpy as np
from matplotlib.animation import FuncAnimation

G = 1.0


class Body:
    def __init__(self, mass, radius, position, velocity):
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def force_from(self, other):
        r = other.position - self.position
        dist = np.linalg.norm(r)
        if dist == 0:
            return np.zeros(2)
        return G * self.mass * other.mass * r / dist**3


def orbital_velocity(center, orbiting, clockwise=True):
    r_vec = orbiting.position - center.position
    r = np.linalg.norm(r_vec)

    if clockwise:
        tang = np.array([r_vec[1], -r_vec[0]]) / r
    else:
        tang = np.array([-r_vec[1], r_vec[0]]) / r

    v = np.sqrt(G * center.mass / r)
    return tang * v


def update(frame, bodies, scat, dt):
    forces = []
    for i, b in enumerate(bodies):
        total_f = np.zeros(2)
        for j, other in enumerate(bodies):
            if i != j:
                total_f += b.force_from(other)
        forces.append(total_f)

    for i, body in enumerate(bodies):
        body.velocity += forces[i] / body.mass * dt
        body.position += body.velocity * dt

    xs = [body.position[0] for body in bodies]
    ys = [body.position[1] for body in bodies]
    scat.set_offsets(np.c_[xs, ys])

    return (scat,)


if __name__ == "__main__":
    sun = Body(700, 0.5, [0, 0], [0, 0])
    earth = Body(10, 0.05, [1.5, 0], [0, 0])
    moon = Body(5, 0.02, [1.4, 0], [0, 0])

    # Otomatik yörünge hızlarını ata
    earth.velocity = orbital_velocity(sun, earth)
    moon.velocity = earth.velocity + orbital_velocity(earth, moon)

    bodies = [sun, earth, moon]

    dt = 0.002

    fig, ax = plt.subplots()
    scat = ax.scatter(
        [b.position[0] for b in bodies],
        [b.position[1] for b in bodies],
        s=[2000 * b.radius for b in bodies],
    )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")

    ani = FuncAnimation(fig, update, fargs=(bodies, scat, dt), interval=10)
    plt.show()

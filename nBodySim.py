import matplotlib.pyplot as plt

plt.style.use("dark_background")
import numpy as np
from matplotlib.animation import FuncAnimation

G = 1.0


class Body:
    def __init__(self, mass, radius, position, velocity, color):
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color

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
    for body in bodies:
        if np.allclose(body.position, [0, 0]):
            body.velocity = np.zeros(2)
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
    sun = Body(2000, 5, [0, 0], [0, 0], "yellow")
    earth = Body(10, 0.05, [1.5, 0], [0, 0], "blue")
    moon = Body(5, 0.02, [1.4, 0], [0, 0], "gray")
    mars = Body(15, 0.03, [2.5, 0], [0, 0], "orange")

    # Otomatik yörünge hızlarını ata
    # ilk hizlari
    earth.velocity = orbital_velocity(sun, earth)
    moon.velocity = (
        earth.velocity + orbital_velocity(earth, moon) + orbital_velocity(earth, mars)
    )
    mars.velocity = (
        mars.velocity + orbital_velocity(sun, mars) + orbital_velocity(earth, mars)
    )

    bodies = [sun, earth, moon, mars]

    dt = 0.001

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 10))
    scat = ax.scatter(
        [b.position[0] for b in bodies],
        [b.position[1] for b in bodies],
        s=[2000 * b.radius for b in bodies],
        c=[b.color for b in bodies],
    )

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.axis("off")  # Eksenlerin tamamını (çizgiler, sayılar, tick'ler) kapatır
    fig.patch.set_facecolor("black")  # Arka planı siyah yap (uzay hissi için)
    ax.set_facecolor("black")  # Grafik alanını da siyah yap
    ani = FuncAnimation(fig, update, fargs=(bodies, scat, dt), interval=10)
    plt.show()

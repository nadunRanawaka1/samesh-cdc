import numpy as np
import torch


def golden_ratio():
    return (1 + np.sqrt(5)) / 2


def tetrahedron():
    return np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ])


def octohedron():
    return np.array([
        [ 1,  0,  0],
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0],
        [ 0, -1,  0],
    ])


def cube():
    return np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1, -1],
        [ 1, -1, -1],
    ])


def icosahedron():
    phi = golden_ratio()
    return np.array([
        [-1,  phi,  0],
        [-1, -phi,  0],
        [ 1,  phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ]) / np.sqrt(1 + phi ** 2)


def dodecahedron():
    phi = golden_ratio()
    a, b = 1 / phi, 1 / (phi * phi)
    return np.array([
        [-a, -a,  b], [ a, -a,  b], [ a,  a,  b], [-a,  a,  b],
        [-a, -a, -b], [ a, -a, -b], [ a,  a, -b], [-a,  a, -b],
        [ b, -a, -a], [ b,  a, -a], [ b,  a,  a], [ b, -a,  a],
        [-b, -a, -a], [-b,  a, -a], [-b,  a,  a], [-b, -a,  a],
        [-a,  b, -a], [ a,  b, -a], [ a,  b,  a], [-a,  b,  a],
    ]) / np.sqrt(a ** 2 + b ** 2)


def icosidodecahedron():
    """
    30 vertices - combination of icosahedron and dodecahedron
    """
    phi = golden_ratio()
    vertices = []
    # Permutations of (0, 0, phi)
    for sign1 in [-1, 1]:
        vertices.append([0, 0, sign1 * phi])
        vertices.append([0, sign1 * phi, 0])
        vertices.append([sign1 * phi, 0, 0])
    # Permutations of (1/2, phi/2, phi^2/2)
    phi2 = phi * phi
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                vertices.append([s1 * 0.5, s2 * phi / 2, s3 * phi2 / 2])
                vertices.append([s1 * phi / 2, s2 * phi2 / 2, s3 * 0.5])
                vertices.append([s1 * phi2 / 2, s2 * 0.5, s3 * phi / 2])
    vertices = np.array(vertices)
    return vertices / np.linalg.norm(vertices[0])


def truncated_octahedron():
    """
    24 vertices - octahedron with corners cut off
    """
    vertices = []
    # All permutations of (0, 1, 2)
    coords = [0, 1, 2]
    from itertools import permutations
    for perm in permutations(coords):
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v = [s1 * perm[0] if perm[0] != 0 else 0,
                     s2 * perm[1] if perm[1] != 0 else 0,
                     perm[2]]
                if perm[2] != 0:
                    for s3 in [-1, 1]:
                        vertices.append([s1 * perm[0] if perm[0] != 0 else 0,
                                        s2 * perm[1] if perm[1] != 0 else 0,
                                        s3 * perm[2]])
                else:
                    vertices.append(v)
    # Remove duplicates
    vertices = np.array(vertices)
    vertices = np.unique(vertices, axis=0)
    return vertices / np.linalg.norm(vertices[0])


def rhombicuboctahedron():
    """
    24 vertices - expanded cube/octahedron
    """
    vertices = []
    # All permutations of (±1, ±1, ±(1+√2))
    r = 1 + np.sqrt(2)
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                vertices.append([s1 * 1, s2 * 1, s3 * r])
                vertices.append([s1 * 1, s2 * r, s3 * 1])
                vertices.append([s1 * r, s2 * 1, s3 * 1])
    vertices = np.array(vertices)
    return vertices / np.linalg.norm(vertices[0])


def truncated_icosahedron():
    """
    60 vertices - soccer ball / buckyball shape
    """
    phi = golden_ratio()
    vertices = []
    
    # Even permutations of (0, 1, 3*phi)
    three_phi = 3 * phi
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            vertices.append([0, s1 * 1, s2 * three_phi])
            vertices.append([s1 * 1, s2 * three_phi, 0])
            vertices.append([s1 * three_phi, 0, s2 * 1])
    
    # Even permutations of (1, 2+phi, 2*phi)
    two_plus_phi = 2 + phi
    two_phi = 2 * phi
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                vertices.append([s1 * 1, s2 * two_plus_phi, s3 * two_phi])
                vertices.append([s1 * two_plus_phi, s2 * two_phi, s3 * 1])
                vertices.append([s1 * two_phi, s2 * 1, s3 * two_plus_phi])
    
    # Even permutations of (phi, 2, phi^3)
    phi3 = phi ** 3
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                vertices.append([s1 * phi, s2 * 2, s3 * phi3])
                vertices.append([s1 * 2, s2 * phi3, s3 * phi])
                vertices.append([s1 * phi3, s2 * phi, s3 * 2])
    
    vertices = np.array(vertices)
    return vertices / np.linalg.norm(vertices[0])


def snub_cube():
    """
    24 vertices - chiral polyhedron with good coverage
    """
    # Tribonacci constant
    t = (1 + np.cbrt(19 + 3*np.sqrt(33)) + np.cbrt(19 - 3*np.sqrt(33))) / 3
    vertices = []
    
    # Even permutations with all sign combinations
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                vertices.append([s1 * 1, s2 * (1/t), s3 * t])
                vertices.append([s1 * (1/t), s2 * t, s3 * 1])
                vertices.append([s1 * t, s2 * 1, s3 * (1/t)])
    
    vertices = np.array(vertices)
    return vertices / np.linalg.norm(vertices[0])


def geodesic_sphere(subdivisions=2):
    """
    Geodesic sphere from subdivided icosahedron.
    subdivisions=1: 12 vertices (icosahedron)
    subdivisions=2: 42 vertices
    subdivisions=3: 162 vertices
    subdivisions=4: 642 vertices
    """
    phi = golden_ratio()
    
    # Start with icosahedron vertices
    verts = [
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1],
    ]
    verts = np.array(verts, dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    
    # Icosahedron faces (triangles)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    
    def midpoint(v1, v2):
        mid = (v1 + v2) / 2
        return mid / np.linalg.norm(mid)
    
    for _ in range(subdivisions - 1):
        new_faces = []
        edge_cache = {}
        
        for tri in faces:
            v = [verts[tri[0]], verts[tri[1]], verts[tri[2]]]
            mids = []
            
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                if edge not in edge_cache:
                    new_vert = midpoint(verts[edge[0]], verts[edge[1]])
                    edge_cache[edge] = len(verts)
                    verts = np.vstack([verts, new_vert])
                mids.append(edge_cache[edge])
            
            # Create 4 new triangles
            new_faces.append([tri[0], mids[0], mids[2]])
            new_faces.append([tri[1], mids[1], mids[0]])
            new_faces.append([tri[2], mids[2], mids[1]])
            new_faces.append([mids[0], mids[1], mids[2]])
        
        faces = new_faces
    
    return verts


def fibonacci_sphere(n=100):
    """
    Generate n approximately uniformly distributed points on a sphere
    using the Fibonacci/golden spiral method.
    """
    phi = golden_ratio()
    indices = np.arange(n)
    
    # Golden angle in radians
    theta = 2 * np.pi * indices / phi
    
    # Z coordinates evenly spaced from -1 to 1
    z = 1 - (2 * indices + 1) / n
    
    # Radius at each z
    radius = np.sqrt(1 - z * z)
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    return np.column_stack([x, y, z])


def standard(n=8, elevation=15):
    """
    """
    pphi =  elevation * np.pi / 180
    nphi = -elevation * np.pi / 180
    coords = []
    for phi in [pphi, nphi]:
        for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
            coords.append([
                np.cos(theta) * np.cos(phi),
                np.sin(phi),
                np.sin(theta) * np.cos(phi),
            ])
    coords.append([0,  0,  1])
    coords.append([0,  0, -1])
    return np.array(coords)


def swirl(n=120, cycles=1, elevation_range=(-45, 60)):
    """
    """
    pphi = elevation_range[0] * np.pi / 180
    nphi = elevation_range[1] * np.pi / 180
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coords = []
    for i, phi in enumerate(np.linspace(pphi, nphi, n)):
        coords.append([
            np.cos(cycles * thetas[i]) * np.cos(phi),
            np.sin(phi),
            np.sin(cycles * thetas[i]) * np.cos(phi),
        ])
    return np.array(coords)
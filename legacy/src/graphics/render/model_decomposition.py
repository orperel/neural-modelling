from panda3d.core import GeomVertexReader
from src.graphics import Mesh


# Reference:
# https://www.panda3d.org/manual/?title=Reading_existing_geometry_data
def processGeomNode(geomNode):
    mesh = Mesh()
    for i in range(geomNode.getNumGeoms()):
        geom = geomNode.getGeom(i)
        state = geomNode.getGeomState(i)
        # print geom
        # print state
        processGeom(geom, mesh)
    return mesh

def processGeom(geom, mesh):
    vdata = geom.getVertexData()
    print(vdata)
    processVertexData(vdata, mesh)
    for i in range(geom.getNumPrimitives()):
        prim = geom.getPrimitive(i)
        print(prim)
        processPrimitive(prim, vdata, mesh)

def processVertexData(vdata, mesh):
    vertex = GeomVertexReader(vdata, 'vertex')
    # texcoord = GeomVertexReader(vdata, 'texcoord')
    while not vertex.isAtEnd():
        v = vertex.getData3f()
        # t = texcoord.getData2f()
        # print "v = %s, t = %s" % (repr(v), repr(t))
        print("v = %s" % (repr(v)))
        mesh.add_vertex((v[0], v[1], v[2]))

def processPrimitive(prim, vdata, mesh):
    vertex = GeomVertexReader(vdata, 'vertex')

    prim = prim.decompose()

    for p in range(prim.getNumPrimitives()):
        s = prim.getPrimitiveStart(p)
        e = prim.getPrimitiveEnd(p)
        vertices = []
        for i in range(s, e):
            vi = prim.getVertex(i)
            vertices.append(vi)
            vertex.setRow(vi)
            v = vertex.getData3f()
            print("prim %s has vertex %s: %s" % (p, vi, repr(v)))
        mesh.add_face(f=vertices)

def parse_model_geometry(model):
    meshes = []
    geomNodeCollection = model.findAllMatches('**/+GeomNode')
    for nodePath in geomNodeCollection:
        geomNode = nodePath.node()
        mesh = processGeomNode(geomNode)
        meshes.append(mesh)
    return meshes

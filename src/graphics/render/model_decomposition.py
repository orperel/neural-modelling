from panda3d.core import GeomVertexReader

# Reference:
# https://www.panda3d.org/manual/?title=Reading_existing_geometry_data

def processGeomNode(geomNode):
  for i in range(geomNode.getNumGeoms()):
    geom = geomNode.getGeom(i)
    state = geomNode.getGeomState(i)
    # print geom
    # print state
    processGeom(geom)

def processGeom(geom):
  vdata = geom.getVertexData()
  print(vdata)
  processVertexData(vdata)
  for i in range(geom.getNumPrimitives()):
      prim = geom.getPrimitive(i)
      print(prim)
      processPrimitive(prim, vdata)

def processVertexData(vdata):
  vertex = GeomVertexReader(vdata, 'vertex')
  # texcoord = GeomVertexReader(vdata, 'texcoord')
  while not vertex.isAtEnd():
    v = vertex.getData3f()
    # t = texcoord.getData2f()
    # print "v = %s, t = %s" % (repr(v), repr(t))
    print("v = %s" % (repr(v)))

def processPrimitive(prim, vdata):
    vertex = GeomVertexReader(vdata, 'vertex')

    prim = prim.decompose()

    for p in range(prim.getNumPrimitives()):
        s = prim.getPrimitiveStart(p)
        e = prim.getPrimitiveEnd(p)
        for i in range(s, e):
            vi = prim.getVertex(i)
            vertex.setRow(vi)
            v = vertex.getData3f()
            print("prim %s has vertex %s: %s" % (p, vi, repr(v)))

def parse_model_geometry(model):
    geomNodeCollection = model.findAllMatches('**/+GeomNode')
    for nodePath in geomNodeCollection:
      geomNode = nodePath.node()
      processGeomNode(geomNode)

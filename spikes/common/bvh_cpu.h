// bvh_cpu.h : a plain-C++ copy of Cycles' BVH2 traversal, line for line.
// Updated for M2: Instance leaf support.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include "bvh_dump.h"

namespace cyc {

constexpr int kEntrySentinel = 0x76543210;
constexpr int kStackSize = 192;
constexpr uint32_t kMaxIterations = 1000000;

constexpr uint32_t kErrLoop = 0xFFFFFFFBu;
constexpr uint32_t kErrStack = 0xFFFFFFFCu;
constexpr uint32_t kErrBadIndex = 0xFFFFFFFDu;
constexpr uint32_t kErrInstance = 0xFFFFFFFEu;
constexpr uint32_t kNoPrim = 0xFFFFFFFFu;

inline bool is_error_id(uint32_t id) { return id >= kErrLoop && id != kNoPrim; }
constexpr uint32_t kPrimTriangle = 1u;

struct V3 { float x, y, z; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross3(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline int as_int(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline uint32_t as_uint(float f) { uint32_t i; std::memcpy(&i, &f, 4); return i; }
inline float as_float(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// --- M2 ADDITION: Transform helpers ---
struct Transform { float x[4], y[4], z[4]; };
inline V3 transform_point(const Transform& t, V3 a) {
  return { a.x * t.x[0] + a.y * t.x[1] + a.z * t.x[2] + t.x[3],
           a.x * t.y[0] + a.y * t.y[1] + a.z * t.y[2] + t.y[3],
           a.x * t.z[0] + a.y * t.z[1] + a.z * t.z[2] + t.z[3] };
}
inline V3 transform_direction(const Transform& t, V3 a) {
  return { a.x * t.x[0] + a.y * t.x[1] + a.z * t.x[2],
           a.x * t.y[0] + a.y * t.y[1] + a.z * t.y[2],
           a.x * t.z[0] + a.y * t.z[1] + a.z * t.z[2] };
}
// --------------------------------------

struct SceneView {
  const float *nodes;
  const float *leaves;
  const uint32_t *prim_index, *prim_visibility, *prim_object, *tri_vindex;
  const float *tri_verts;
  const uint32_t *objects;
  const uint32_t *object_node; // <-- M2 ADDITION
  int32_t root;
  uint32_t node_count, leaf_count, prim_count, tri_count, vert_count, object_count, object_node_count; // <-- UPDATED
};

inline SceneView make_view(const BvhDump &d) {
  SceneView v{};
  v.nodes = d.nodes.data(); v.leaves = d.leaves.data();
  v.prim_index = d.prim_index.data(); v.prim_visibility = d.prim_visibility.data();
  v.prim_object = d.prim_object.data(); v.tri_vindex = d.tri_vindex.data();
  v.tri_verts = d.tri_verts.data(); v.objects = d.objects.data();
  v.object_node = d.object_node.data(); // <-- M2 ADDITION
  v.root = d.root(); v.node_count = d.node_count(); v.leaf_count = d.leaf_count();
  v.prim_count = d.prim_count(); v.tri_count = d.tri_count();
  v.vert_count = d.vert_count(); v.object_count = d.object_count();
  v.object_node_count = d.object_node_count(); // <-- M2 ADDITION
  return v;
}

inline float clamp_component(float c) {
  const float ooeps = 8.271806E-25f;
  if (std::fabs(c) > ooeps) return c;
  return std::copysign(ooeps, c);
}

inline bool ray_triangle_intersect(V3 P, V3 D, float tmin, float tmax, V3 a, V3 b, V3 c, float &ou, float &ov, float &ot) {
  const V3 v0 = a - P, v1 = b - P, v2 = c - P;
  const V3 e0 = v2 - v0, e1 = v0 - v1, e2 = v1 - v2;
  const float U = dot3(cross3(e0, v2 + v0), D);
  const float V = dot3(cross3(e1, v0 + v1), D);
  const float W = dot3(cross3(e2, v1 + v2), D);
  const float UVW = U + V + W;
  const float eps = 1.1920929e-07f * std::fabs(UVW);
  const float minUVW = std::fmin(U, std::fmin(V, W));
  const float maxUVW = std::fmax(U, std::fmax(V, W));
  if (!(minUVW >= -eps || maxUVW <= eps)) return false;
  const V3 Ng1 = cross3(e1, e0);
  const V3 Ng = Ng1 + Ng1;
  const float den = dot3(Ng, D);
  if (den == 0.0f) return false;
  const float T = dot3(v0, Ng);
  const float t = T / den;
  if (!(t >= tmin && t <= tmax)) return false;
  const float rcp_uvw = (std::fabs(UVW) < 1e-18f) ? 0.0f : 1.0f / UVW;
  ou = std::fmin(U * rcp_uvw, 1.0f);
  ov = std::fmin(V * rcp_uvw, 1.0f);
  ot = t;
  return true;
}

struct F4 { float x, y, z, w; };
inline F4 fetch4(const float *base, uint32_t i) {
  return {base[i * 4 + 0], base[i * 4 + 1], base[i * 4 + 2], base[i * 4 + 3]};
}

inline uint32_t traverse(const SceneView &sc, V3 P, V3 Din, float tmin, float tmax,
                         uint32_t visibility, float &hit_t, uint32_t &hit_prim, uint32_t &hit_object) {
  int stack[kStackSize];
  stack[0] = kEntrySentinel;
  int sp = 0;
  int node_addr = sc.root;

  // --- M2 ADDITION: Save original ray state for instance pop ---
  V3 orig_P = P;
  V3 orig_Din = Din;
  float orig_tmin = tmin;
  float orig_tmax = tmax;
  uint32_t current_object = 0xFFFFFFFFu; // OBJECT_NONE
  // -------------------------------------------------------------

  V3 dir = {clamp_component(Din.x), clamp_component(Din.y), clamp_component(Din.z)};
  V3 idir = {1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z};

  hit_t = tmax;
  hit_prim = kNoPrim;
  hit_object = 0xFFFFFFFFu; // <-- ADDED: Initialize to OBJECT_NONE on miss
  uint32_t iterations = 0;

  do {
    while (node_addr >= 0 && node_addr != kEntrySentinel) {
      if (++iterations > kMaxIterations) return kErrLoop;
      if ((uint32_t)node_addr + 3u >= sc.node_count) return kErrBadIndex;

      const F4 cnodes = fetch4(sc.nodes, (uint32_t)node_addr + 0);
      const F4 node0 = fetch4(sc.nodes, (uint32_t)node_addr + 1);
      const F4 node1 = fetch4(sc.nodes, (uint32_t)node_addr + 2);
      const F4 node2 = fetch4(sc.nodes, (uint32_t)node_addr + 3);

      const float c0lox = (node0.x - P.x) * idir.x;
      const float c0hix = (node0.z - P.x) * idir.x;
      const float c0loy = (node1.x - P.y) * idir.y;
      const float c0hiy = (node1.z - P.y) * idir.y;
      const float c0loz = (node2.x - P.z) * idir.z;
      const float c0hiz = (node2.z - P.z) * idir.z;
      const float c0min = std::fmax(std::fmax(tmin, std::fmin(c0lox, c0hix)), std::fmax(std::fmin(c0loy, c0hiy), std::fmin(c0loz, c0hiz)));
      const float c0max = std::fmin(std::fmin(hit_t, std::fmax(c0lox, c0hix)), std::fmin(std::fmax(c0loy, c0hiy), std::fmax(c0loz, c0hiz)));

      const float c1lox = (node0.y - P.x) * idir.x;
      const float c1hix = (node0.w - P.x) * idir.x;
      const float c1loy = (node1.y - P.y) * idir.y;
      const float c1hiy = (node1.w - P.y) * idir.y;
      const float c1loz = (node2.y - P.z) * idir.z;
      const float c1hiz = (node2.w - P.z) * idir.z;
      const float c1min = std::fmax(std::fmax(tmin, std::fmin(c1lox, c1hix)), std::fmax(std::fmin(c1loy, c1hiy), std::fmin(c1loz, c1hiz)));
      const float c1max = std::fmin(std::fmin(hit_t, std::fmax(c1lox, c1hix)), std::fmin(std::fmax(c1loy, c1hiy), std::fmax(c1loz, c1hiz)));

      const float dist0 = c0min;
      const float dist1 = c1min;

      const int traverse_mask = (((c0max >= c0min) && (as_uint(cnodes.x) & visibility)) ? 1 : 0) |
                                (((c1max >= c1min) && (as_uint(cnodes.y) & visibility)) ? 2 : 0);

      node_addr = as_int(cnodes.z);
      int node_addr_child1 = as_int(cnodes.w);

      if (traverse_mask == 3) {
        if (dist1 < dist0) { const int tmp = node_addr; node_addr = node_addr_child1; node_addr_child1 = tmp; }
        ++sp;
        if (sp >= kStackSize) return kErrStack;
        stack[sp] = node_addr_child1;
      } else {
        if (traverse_mask == 2) { node_addr = node_addr_child1; }
        else if (traverse_mask == 0) {
          if (sp < 0) return kErrStack;
          node_addr = stack[sp];
          --sp;
        }
      }
    }

    if (node_addr < 0) {
      const int leaf_index = -node_addr - 1;
      if ((uint32_t)leaf_index >= sc.leaf_count) return kErrBadIndex;
      const F4 leaf = fetch4(sc.leaves, (uint32_t)leaf_index);
      int prim_addr = as_int(leaf.x);

      if (prim_addr >= 0) {
        const int prim_addr2 = as_int(leaf.y);
        const uint32_t type = as_uint(leaf.w);
        if (sp < 0) return kErrStack;
        node_addr = stack[sp];
        --sp;
        if (prim_addr2 < prim_addr || (uint32_t)prim_addr2 > sc.prim_count) return kErrBadIndex;

        for (; prim_addr < prim_addr2; prim_addr++) {
          // --- M2 ADDITION: Use instance object ID if we are inside an instance ---
          const uint32_t pobj = (current_object != 0xFFFFFFFFu) ? current_object : sc.prim_object[prim_addr];
          const uint32_t prim = sc.prim_index[prim_addr];

          if (type & kPrimTriangle) {
            if (pobj >= sc.object_count || prim >= sc.tri_count) return kErrBadIndex;
            const uint32_t pos_off = sc.objects[(size_t)pobj * 64u + 42u]; // 42 is kPositionOffsetWord
            const uint32_t ia = sc.tri_vindex[prim * 3 + 0] + pos_off;
            const uint32_t ib = sc.tri_vindex[prim * 3 + 1] + pos_off;
            const uint32_t ic = sc.tri_vindex[prim * 3 + 2] + pos_off;
            if (ia >= sc.vert_count || ib >= sc.vert_count || ic >= sc.vert_count) return kErrBadIndex;

            const V3 a = {sc.tri_verts[ia * 3 + 0], sc.tri_verts[ia * 3 + 1], sc.tri_verts[ia * 3 + 2]};
            const V3 b = {sc.tri_verts[ib * 3 + 0], sc.tri_verts[ib * 3 + 1], sc.tri_verts[ib * 3 + 2]};
            const V3 c = {sc.tri_verts[ic * 3 + 0], sc.tri_verts[ic * 3 + 1], sc.tri_verts[ic * 3 + 2]};

          float u, v, t;
            if (ray_triangle_intersect(P, dir, tmin, hit_t, a, b, c, u, v, t)) {
              if (sc.prim_visibility[prim_addr] & visibility) {
                hit_t = t;
                hit_prim = prim;
                hit_object = pobj; // <-- ADDED: Record which object was hit
              }
            }
          } else {
            return kErrInstance;
          }
        }
      } else {
        // --- M2 ADDITION: INSTANCE LEAF PUSH ---
        int instance_idx = -prim_addr - 1;
        if ((uint32_t)instance_idx >= sc.prim_count) return kErrBadIndex;
        uint32_t obj = sc.prim_object[instance_idx];
        if (obj >= sc.object_count) return kErrBadIndex;
        if (sc.object_node[obj] >= sc.node_count) return kErrBadIndex;

        // Fetch inverse transform (words 12-23 of the 64-word KernelObject)
        const uint32_t* obj_data = &sc.objects[obj * 64u];
        Transform itfm;
        itfm.x[0] = as_float(obj_data[12]); itfm.x[1] = as_float(obj_data[13]); itfm.x[2] = as_float(obj_data[14]); itfm.x[3] = as_float(obj_data[15]);
        itfm.y[0] = as_float(obj_data[16]); itfm.y[1] = as_float(obj_data[17]); itfm.y[2] = as_float(obj_data[18]); itfm.y[3] = as_float(obj_data[19]);
        itfm.z[0] = as_float(obj_data[20]); itfm.z[1] = as_float(obj_data[21]); itfm.z[2] = as_float(obj_data[22]); itfm.z[3] = as_float(obj_data[23]);

        // bvh_instance_push
        P = transform_point(itfm, P);
        V3 new_dir = transform_direction(itfm, dir);
        
        // Update ray state for the instance traversal (now safe because 'dir' is not const)
        dir = {clamp_component(new_dir.x), clamp_component(new_dir.y), clamp_component(new_dir.z)};
        idir = {1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z};

        // Push sentinel and new node address
        ++sp;
        if (sp >= kStackSize) return kErrStack;
        stack[sp] = kEntrySentinel;

        current_object = obj;
        node_addr = sc.object_node[obj];
      }
    }
    
     // --- M2 ADDITION: INSTANCE POP ---
    if (node_addr == kEntrySentinel) {
      P = orig_P;
      V3 restored_dir = {clamp_component(orig_Din.x), clamp_component(orig_Din.y), clamp_component(orig_Din.z)};
      dir = restored_dir;
      idir = {1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z};
      tmin = orig_tmin;
      tmax = orig_tmax;
      current_object = 0xFFFFFFFFu;

      // Only pop again if there is actually something left on the stack.
      // If sp < 0, we just popped the final sentinel and the traversal is done.
      if (sp >= 0) {
        node_addr = stack[sp];
        --sp;
      }
    }
  } while (node_addr != kEntrySentinel);

  return 0;
}
} // namespace cyc
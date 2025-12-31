//! Catmull-Clark subdivision surfaces via analytical limit evaluation.
//!
//! Instead of recursive tessellation, we evaluate the limit surface directly
//! using Stam's 1998 eigenanalysis. For a face with vertices of valence V,
//! the limit position is:
//!
//! ```text
//! P(u,v) = Σᵢ λᵢⁿ⁻¹ · φᵢ(u,v) · cᵢ
//! ```
//!
//! Where:
//! - λᵢ are eigenvalues of the subdivision matrix
//! - φᵢ(u,v) are bicubic basis functions
//! - cᵢ are control point coefficients (linear combination of cage vertices)
//! - n is the subdivision level (for limit surface, n → ∞, so λᵢⁿ⁻¹ vanishes for |λᵢ| < 1)
//!
//! Derivatives come for free via Jet3:
//! - Evaluate with Jet3<Field> instead of Field
//! - Normal = cross(dP/du, dP/dv)
//! - No finite differences, no extra evaluations

use crate::mesh::{Point3, QuadMesh};

/// Eigenstructure for a given valence configuration.
///
/// Precomputed at compile time via const fn (future work).
/// For now, tables will be baked from Stam's formulas.
#[derive(Clone, Debug)]
pub struct EigenStructure {
    /// Valence (number of edges meeting at extraordinary vertex)
    pub valence: usize,
    /// Eigenvalues λᵢ of subdivision matrix
    pub eigenvalues: Vec<f32>,
    /// Eigenvectors (basis for limit surface evaluation)
    pub eigenvectors: Vec<Vec<f32>>,
}

impl EigenStructure {
    /// Create eigenstructure for regular vertex (valence 4).
    ///
    /// This is the simplest case - reduces to bicubic B-spline.
    pub fn regular() -> Self {
        Self {
            valence: 4,
            eigenvalues: vec![1.0, 0.25, 0.25, 0.0625],
            eigenvectors: Vec::new(), // TODO: B-spline basis
        }
    }

    /// Create eigenstructure for arbitrary valence.
    ///
    /// Uses Stam's formulas for eigendecomposition.
    pub fn for_valence(valence: usize) -> Self {
        // TODO: Implement Stam eigenanalysis
        // For now, return placeholder
        Self {
            valence,
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
        }
    }
}

/// A subdivision patch - one quad face from the control cage.
///
/// Stores the indices and local topology needed to evaluate
/// the limit surface at (u,v) ∈ [0,1]².
#[derive(Clone, Debug)]
pub struct SubdivisionPatch {
    /// Index of this face in the mesh
    pub face_idx: usize,
    /// The 4 corner vertex indices [v0, v1, v2, v3]
    pub corners: [usize; 4],
    /// Valences of the 4 corners
    pub corner_valences: [usize; 4],
}

impl SubdivisionPatch {
    /// Extract patch from mesh at given face index.
    pub fn from_mesh(mesh: &QuadMesh, face_idx: usize) -> Result<Self, String> {
        if face_idx >= mesh.face_count() {
            return Err(format!(
                "Face index {} out of bounds (mesh has {} faces)",
                face_idx,
                mesh.face_count()
            ));
        }

        let face = &mesh.faces[face_idx];
        let corners = face.vertices;

        // Validate all vertex indices
        for &v in &corners {
            if !mesh.is_valid_vertex(v) {
                return Err(format!("Invalid vertex index {} in face {}", v, face_idx));
            }
        }

        let corner_valences = [
            mesh.valence[corners[0]],
            mesh.valence[corners[1]],
            mesh.valence[corners[2]],
            mesh.valence[corners[3]],
        ];

        Ok(Self {
            face_idx,
            corners,
            corner_valences,
        })
    }

    /// Check if this patch has any extraordinary vertices.
    ///
    /// Extraordinary = valence != 4 (the regular case).
    pub fn is_extraordinary(&self) -> bool {
        self.corner_valences.iter().any(|&v| v != 4)
    }

    /// Get maximum valence among corners.
    pub fn max_valence(&self) -> usize {
        *self.corner_valences.iter().max().unwrap()
    }

    /// Evaluate limit surface at (u,v).
    ///
    /// For now, returns the bilinear interpolation of corners.
    /// TODO: Replace with Stam eigenanalysis.
    pub fn eval(&self, mesh: &QuadMesh, u: f32, v: f32) -> Point3 {
        // Bilinear interpolation as placeholder
        let p0 = mesh.vertices[self.corners[0]];
        let p1 = mesh.vertices[self.corners[1]];
        let p2 = mesh.vertices[self.corners[2]];
        let p3 = mesh.vertices[self.corners[3]];

        let u1 = 1.0 - u;
        let v1 = 1.0 - v;

        Point3::new(
            u1 * v1 * p0.x + u * v1 * p1.x + u * v * p2.x + u1 * v * p3.x,
            u1 * v1 * p0.y + u * v1 * p1.y + u * v * p2.y + u1 * v * p3.y,
            u1 * v1 * p0.z + u * v1 * p1.z + u * v * p2.z + u1 * v * p3.z,
        )
    }
}

/// Subdivision surface - collection of patches with shared topology.
#[derive(Clone, Debug)]
pub struct SubdivisionSurface {
    /// The control cage mesh
    pub mesh: QuadMesh,
    /// One patch per face
    pub patches: Vec<SubdivisionPatch>,
}

impl SubdivisionSurface {
    /// Build subdivision surface from quad mesh.
    pub fn from_mesh(mesh: QuadMesh) -> Result<Self, String> {
        let mut patches = Vec::with_capacity(mesh.face_count());

        for face_idx in 0..mesh.face_count() {
            patches.push(SubdivisionPatch::from_mesh(&mesh, face_idx)?);
        }

        Ok(Self { mesh, patches })
    }

    /// Get number of patches.
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Get statistics about extraordinary vertices.
    pub fn stats(&self) -> SurfaceStats {
        let total_patches = self.patches.len();
        let extraordinary_patches = self.patches.iter().filter(|p| p.is_extraordinary()).count();
        let max_valence = self.patches.iter().map(|p| p.max_valence()).max().unwrap_or(0);

        SurfaceStats {
            total_patches,
            extraordinary_patches,
            max_valence,
        }
    }
}

/// Statistics about a subdivision surface.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceStats {
    pub total_patches: usize,
    pub extraordinary_patches: usize,
    pub max_valence: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::io::BufReader;

    #[test]
    fn test_regular_patch() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        let patch = SubdivisionPatch::from_mesh(&mesh, 0).unwrap();

        // All corners have valence 1 (only one face)
        assert_eq!(patch.corner_valences, [1, 1, 1, 1]);
        assert!(patch.is_extraordinary()); // Valence 1 is extraordinary
    }

    #[test]
    fn test_bilinear_eval() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        let patch = SubdivisionPatch::from_mesh(&mesh, 0).unwrap();

        // Center should be (0.5, 0.5, 0.0)
        let center = patch.eval(&mesh, 0.5, 0.5);
        assert!((center.x - 0.5).abs() < 1e-6);
        assert!((center.y - 0.5).abs() < 1e-6);
        assert!(center.z.abs() < 1e-6);
    }

    #[test]
    fn test_surface_stats() {
        let obj = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 2.0 0.0 0.0
v 2.0 1.0 0.0
f 1 2 3 4
f 2 5 6 3
";
        let mesh = QuadMesh::from_obj(BufReader::new(Cursor::new(obj))).unwrap();
        let surface = SubdivisionSurface::from_mesh(mesh).unwrap();
        let stats = surface.stats();

        assert_eq!(stats.total_patches, 2);
    }
}

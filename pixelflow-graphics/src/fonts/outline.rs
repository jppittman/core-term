//! A glyph as geometry: closed contours of line and quadratic segments, in
//! whatever frame the caller asked for.
//!
//! Parsing a TrueType glyph produces an [`Outline`]; a rasterizer consumes
//! one. Keeping the two apart is what lets every affine map in a font — a
//! compound glyph's component placement, the em-square scale, the screen
//! flip — be applied to control points on the host, so that a coverage
//! kernel is built in the frame it will be evaluated in and its constants
//! are that frame's numbers. (It used to be the other way round: segments
//! became kernels in a normalized unit square and every transform was a
//! coordinate warp on the finished kernel.) Quadratics are closed under
//! affine maps, so nothing is lost by transforming the control points.

/// A point in the outline's frame: `[x, y]`.
pub type Point = [f32; 2];

/// One piece of a contour, with on-curve endpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Segment {
    /// A straight edge.
    Line {
        /// Where it starts.
        from: Point,
        /// Where it ends.
        to: Point,
    },
    /// A quadratic Bézier: `control` is the off-curve point.
    Quad {
        /// Where it starts.
        from: Point,
        /// The off-curve control point.
        control: Point,
        /// Where it ends.
        to: Point,
    },
}

impl Segment {
    /// The on-curve point the segment starts at.
    #[must_use]
    pub fn from(self) -> Point {
        match self {
            Self::Line { from, .. } | Self::Quad { from, .. } => from,
        }
    }

    /// The on-curve point the segment ends at.
    #[must_use]
    pub fn to(self) -> Point {
        match self {
            Self::Line { to, .. } | Self::Quad { to, .. } => to,
        }
    }

    /// Every control point, in order — the polygon the segment lies inside.
    #[must_use]
    pub fn control_points(self) -> Vec<Point> {
        match self {
            Self::Line { from, to } => vec![from, to],
            Self::Quad { from, control, to } => vec![from, control, to],
        }
    }

    /// The segment's control points pushed through `m`.
    #[must_use]
    pub fn transformed(self, m: Affine) -> Self {
        match self {
            Self::Line { from, to } => Self::Line {
                from: m.apply(from),
                to: m.apply(to),
            },
            Self::Quad { from, control, to } => Self::Quad {
                from: m.apply(from),
                control: m.apply(control),
                to: m.apply(to),
            },
        }
    }
}

/// A closed contour: each segment starts where the previous one ends, and
/// the last ends where the first starts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contour {
    /// The segments, in the direction the font draws them.
    pub segments: Vec<Segment>,
}

impl Contour {
    /// A contour from TrueType's point list — `(x, y, on_curve)` in order —
    /// with the format's implicit on-curve points made explicit: two
    /// consecutive off-curve points imply an on-curve point at their midpoint.
    /// The contour starts at its first on-curve point, so every segment has
    /// on-curve endpoints.
    #[must_use]
    pub fn from_truetype_points(points: &[(f32, f32, bool)]) -> Self {
        let expanded: Vec<(Point, bool)> = points
            .iter()
            .enumerate()
            .flat_map(|(i, &(x, y, on))| {
                let (nx, ny, next_on) = points[(i + 1) % points.len()];
                if !on && !next_on {
                    vec![([x, y], on), ([(x + nx) / 2.0, (y + ny) / 2.0], true)]
                } else {
                    vec![([x, y], on)]
                }
            })
            .collect();
        let Some(start) = expanded.iter().position(|p| p.1) else {
            return Self::default();
        };
        let at = |j: usize| expanded[(start + j) % expanded.len()];
        let mut segments = Vec::with_capacity(expanded.len());
        let mut i = 0;
        while i < expanded.len() {
            let (from, _) = at(i);
            let (next, next_on) = at(i + 1);
            if next_on {
                segments.push(Segment::Line { from, to: next });
                i += 1;
            } else {
                let (to, _) = at(i + 2);
                segments.push(Segment::Quad {
                    from,
                    control: next,
                    to,
                });
                i += 2;
            }
        }
        Self { segments }
    }
}

/// A glyph's geometry: every contour, in one frame.
///
/// Contours may overlap, self-intersect, nest, and run in either direction;
/// what an outline *means* is its winding number under the non-zero rule,
/// which is what every consumer computes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outline {
    /// The contours.
    pub contours: Vec<Contour>,
}

impl Outline {
    /// Whether there is nothing to draw.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contours.iter().all(|c| c.segments.is_empty())
    }

    /// Every segment of every contour.
    pub fn segments(&self) -> impl Iterator<Item = Segment> + '_ {
        self.contours
            .iter()
            .flat_map(|c| c.segments.iter().copied())
    }

    /// The outline pushed through `m`.
    #[must_use]
    pub fn transformed(&self, m: Affine) -> Self {
        Self {
            contours: self
                .contours
                .iter()
                .map(|c| Contour {
                    segments: c.segments.iter().map(|s| s.transformed(m)).collect(),
                })
                .collect(),
        }
    }

    /// The outline translated by `[dx, dy]`.
    #[must_use]
    pub fn translated(&self, [dx, dy]: Point) -> Self {
        self.transformed(Affine::translation(dx, dy))
    }

    /// Every contour of `other`, appended — the outline of a string is the
    /// outlines of its glyphs, placed.
    pub fn append(&mut self, other: Outline) {
        self.contours.extend(other.contours);
    }

    /// The box `[x0, y0, x1, y1]` containing every control point, which
    /// contains every curve. `None` for an empty outline.
    #[must_use]
    pub fn bounds(&self) -> Option<[f32; 4]> {
        let mut out = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let mut any = false;
        for [x, y] in self.segments().flat_map(Segment::control_points) {
            any = true;
            out[0] = out[0].min(x);
            out[1] = out[1].min(y);
            out[2] = out[2].max(x);
            out[3] = out[3].max(y);
        }
        any.then_some(out)
    }
}

/// The forward affine map `x' = a·x + b·y + tx, y' = c·x + d·y + ty`, stored
/// as `[a, b, c, d, tx, ty]` — TrueType's component-transform layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f32; 6]);

impl Affine {
    /// The map that changes nothing.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    /// A pure translation.
    #[must_use]
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self([1.0, 0.0, 0.0, 1.0, tx, ty])
    }

    /// `m(p)`.
    #[must_use]
    pub fn apply(self, [x, y]: Point) -> Point {
        let [a, b, c, d, tx, ty] = self.0;
        [a * x + b * y + tx, c * x + d * y + ty]
    }

    /// The map that applies `self` first and then `outer`: `outer ∘ self`.
    #[must_use]
    pub fn then(self, outer: Self) -> Self {
        let [a, b, c, d, tx, ty] = self.0;
        let [oa, ob, oc, od, otx, oty] = outer.0;
        Self([
            oa * a + ob * c,
            oa * b + ob * d,
            oc * a + od * c,
            oc * b + od * d,
            oa * tx + ob * ty + otx,
            oc * tx + od * ty + oty,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every contour a font produces is closed, segment to segment: that is
    /// the property every winding computation depends on, and the implicit
    /// midpoint expansion is where it would break.
    fn assert_closed(contour: &Contour) {
        let n = contour.segments.len();
        for (i, s) in contour.segments.iter().enumerate() {
            let next = contour.segments[(i + 1) % n];
            assert_eq!(
                s.to(),
                next.from(),
                "segment {i} does not meet segment {}",
                (i + 1) % n
            );
        }
    }

    #[test]
    fn all_on_curve_points_make_lines() {
        let c = Contour::from_truetype_points(&[
            (0.0, 0.0, true),
            (4.0, 0.0, true),
            (4.0, 4.0, true),
            (0.0, 4.0, true),
        ]);
        assert_eq!(c.segments.len(), 4);
        assert!(c.segments.iter().all(|s| matches!(s, Segment::Line { .. })));
        assert_closed(&c);
    }

    #[test]
    fn consecutive_off_curve_points_imply_a_midpoint() {
        // A circle-ish contour of four off-curve points and no on-curve ones:
        // four implied midpoints, four quadratics, starting at a midpoint.
        let c = Contour::from_truetype_points(&[
            (1.0, 0.0, false),
            (1.0, 1.0, false),
            (0.0, 1.0, false),
            (0.0, 0.0, false),
        ]);
        assert_eq!(c.segments.len(), 4);
        assert!(c.segments.iter().all(|s| matches!(s, Segment::Quad { .. })));
        assert_eq!(c.segments[0].from(), [1.0, 0.5]);
        assert_closed(&c);
    }

    #[test]
    fn a_contour_starting_off_curve_is_rotated_to_an_on_curve_start() {
        let c =
            Contour::from_truetype_points(&[(0.0, 1.0, false), (1.0, 0.0, true), (2.0, 2.0, true)]);
        assert_eq!(c.segments.len(), 2);
        assert_eq!(c.segments[0].from(), [1.0, 0.0]);
        assert!(matches!(c.segments[1], Segment::Quad { .. }));
        assert_closed(&c);
    }

    #[test]
    fn a_single_off_curve_point_is_a_degenerate_quad() {
        let c = Contour::from_truetype_points(&[(3.0, 3.0, false)]);
        assert_eq!(c.segments.len(), 1);
        assert_eq!(
            c.segments[0],
            Segment::Quad {
                from: [3.0, 3.0],
                control: [3.0, 3.0],
                to: [3.0, 3.0]
            }
        );
    }

    #[test]
    fn affine_composition_applies_left_to_right() {
        let scale = Affine([2.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
        let shift = Affine::translation(1.0, -1.0);
        let p = [1.0, 1.0];
        assert_eq!(scale.then(shift).apply(p), shift.apply(scale.apply(p)));
        assert_eq!(scale.then(shift).apply(p), [3.0, 2.0]);
        assert_eq!(Affine::IDENTITY.then(scale), scale);
    }

    #[test]
    fn bounds_cover_control_points() {
        let mut o = Outline::default();
        o.contours.push(Contour {
            segments: vec![Segment::Quad {
                from: [0.0, 0.0],
                control: [5.0, -2.0],
                to: [1.0, 1.0],
            }],
        });
        assert_eq!(o.bounds(), Some([0.0, -2.0, 5.0, 1.0]));
        assert_eq!(Outline::default().bounds(), None);
        assert!(Outline::default().is_empty());
    }
}

% scale(1000) import("shank.stl");

// Sketch PureShapes 210
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 105.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 210.000000;
translate([0, 0, -thickness]) {
  translate([-15.000000, -15.000000, 0]) {
    rotate([0, 0, 0.0]) {
      cube([30.000000, 30.000000, thickness]);
    }
  }
}
}

% scale(1000) import("part_1.stl");

// Sketch PureShapes 200
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -99.99999999999999], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 200.000000;
translate([0, 0, -thickness]) {
  translate([-50.000000, -40.000000, 0]) {
    rotate([0, 0, 0.0]) {
      cube([60.000000, 80.000000, thickness]);
    }
  }
}
}

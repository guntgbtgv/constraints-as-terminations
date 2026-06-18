% scale(1000) import("actuator.stl");

// Sketch PureShapes 40
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 20.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 40.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=50.000000,h=thickness);
  }
}
}

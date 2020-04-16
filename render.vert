#version 300 es

varying vec2 uVu;

void main() {

uVu = uv;
gl_Position = vec4(position,1.0);

}

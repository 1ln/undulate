const vert = `
#version 300 es

varying vec2 uVu;

void main() {

uVu = uv;
gl_Position = vec4(position,1.);

}
`;

const frag = `

#version 300 es     

// dolson,2019

out vec4 out_FragColor;

varying vec2 uVu;

uniform vec2 res;

uniform int seed;

uniform float time;

uniform int steps;
uniform float eps;
uniform float dmin;
uniform float dmax;

const float PI   =  radians(180.0); 

float hash(float p) {
    uvec2 n = uint(int(p)) * uvec2(uint(int(seed)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(seed));
    return float(h) * (1./float(0xffffffffU));
}

float hash(vec2 p) {
    uvec2 n = uvec2(ivec2(p)) * uvec2(uint(int(seed)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(seed));
    return float(h) * (1./float(0xffffffffU));
}

float n3(vec3 x) {

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(hash(  n +   0.0) , hash(   n +   1.0)  ,f.x),
                   mix(hash(  n + 157.0) , hash(   n + 158.0)   ,f.x),f.y),
               mix(mix(hash(  n + 113.0) , hash(   n + 114.0)   ,f.x),
                   mix(hash(  n + 270.0) , hash(   n + 271.0)   ,f.x),f.y),f.z);
}

float f3(vec3 x,float hurst) {

    float s = 0.;
    float h = exp2(-hurst);
    float f = 1.;
    float a = .5;

    for(int i = 0; i < 5; i++) {

        s += a * n3(f * x);  
        f *= 2.;
        a *= h;
    }
    return s;
}

float sin3(vec3 p,float s) {
    return sin(p.x * s) * sin(p.y * s) * sin(p.z * s);
}

float envImp(float x,float k) {

    float h = k * x;
    return h * exp(1.0 - h);
}

float sincPh(float x,float k) {

    float a = PI * (k * x - 1.0);
    return sin(a)/a;

}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    
    return a + b * cos( (PI*2.0) * (c * t + d));
}

float easeInOut4(float t) {

    if((t *= 2.0) < 1.0) {
        return 0.5 * t * t;
    } else {
        return -0.5 * ((t - 1.0) * (t - 3.0) - 1.0);
    }
}

float easeOut3(float t) {

    return (t = t - 1.0) * t * t + 1.0;

}

mat2 rot2(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

vec2 opu(vec2 d1,vec2 d2) {

    return (d1.x < d2.x) ? d1 : d2;
} 

float opu(float d1,float d2) {
    
    return min(d1,d2);
}

float opi(float d1,float d2) {

    return max(d1,d2);
}

float opd(float d1,float d2) {

    return max(-d1,d2);
}

float smou(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) - k * h * (1.0 - h);
}

float smod(float d1,float d2,float k) {

    float h = clamp(0.5 - 0.5 * (d2+d1)/k,0.0,1.0);
    return mix(d2,-d1,h) + k * h * (1.0 - h);
}

float smoi(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) + k * h * (1.0 - h);

}

float extr(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
} 

vec2 rev(vec3 p,float w,float f) {
    return vec2(length(p.xz) - w * f,p.y);
} 

vec3 twist(vec3 p,float k) {
    
    float s = sin(k * p.y);
    float c = cos(k * p.y);
    mat2 m = mat2(c,-s,s,c);
    return vec3(m * p.xz,p.y);
}

float circle(vec2 p,float r) {
    return length(p) - r;
}

float sphere(vec3 p,float r) { 
     
    return length(p) - r;
}

float plane(vec3 p,vec4 n) {

    return dot(p,n.xyz) + n.w;
}

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

vec2 scene(vec3 p) {

    vec2 res = vec2(1.,0.);

    float d = 0.;     
    float s = 0.0009;

    float t = time;  
    
    vec3 q = p;
    vec3 l = p;

    p.xz *= rot2(easeOut3(t*s)*0.002);
    q.zy *= rot2(easeInOut4(t*s)*0.02);

    d = mix(sphere(p,0.25),box(q,vec3(1.)),
    sin(s*t)*0.5+0.5); 

    d += n3(p+n3(p)*0.25+t*s)*0.25; 

    res = opu(res,vec2(d,2.)); 

    float pl = plane(l+vec3(0.,1.5,0.),vec4(0.,1.,0.,1.));
    res = opu(res,vec2(pl,1.));
  
  return res;

}

vec2 rayScene(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = dmin;
    float e = dmax;  

    for(int i = 0; i < steps; i++) {

        vec3 p = ro + s * rd;
        vec2 dist = scene(p);
   
        if(abs(dist.x) < eps || e <  dist.x ) { break; }
        s += dist.x;
        d = dist.y;

        }
 
        if(e < s) { d = -1.0; }
        return vec2(s,d);

}

float shadow(vec3 ro,vec3 rd ) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < 100; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,45. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < eps || t > 16.) { break; }

        }

        return clamp(res,0.0,1.0);

}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.0,-1.0) * eps;

    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)).x +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)).x +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)).x + 
    vec3(e.x,e.x,e.x) * scene(p + vec3(e.x,e.x,e.x)).x

    ));
    
}

vec3 rayCamDir(vec2 uv,vec3 camPosition,vec3 camTarget,float fPersp) {

     vec3 camForward = normalize(camTarget - camPosition);
     vec3 camRight = normalize(cross(vec3(0.0,1.0,0.0),camForward));
     vec3 camUp = normalize(cross(camForward,camRight));


     vec3 vDir = normalize(uv.x * camRight + uv.y * camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd) {
 
vec2 d = rayScene(ro, rd);

vec3 col = vec3(1.) - max(rd.y,0.);

if(d.y >= 0.) { 

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(vec3(0.,10.,10.));
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));
float dif = clamp(dot(n,l),0.0,1.0);

float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
* dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);

vec3 linear = vec3(0.);

dif *= shadow(p,l);
ref *= shadow(p,r);

if(d.y == 1.) {

    linear += dif * vec3(.5);
    linear += amb * vec3(.05);
    linear += ref * vec3(4.);
    linear += fre * vec3(.25);

    col = vec3(0.5);

}

if(d.y == 2.) {

    linear += dif * vec3(hash(135.),hash(34.),hash(344.));
    linear += amb * vec3(hash(295.),hash(363.),hash(324.));
    linear += ref * vec3(hash(245.),hash(123.),hash(335.));
    linear += fre * vec3(hash(126.),hash(45.),hash(646.)); 

    float nl = f3(p,hash(122.)); 

    col += fmCol(p.y + nl,vec3(hash(112.),hash(33.),hash(21.)),
                          vec3(hash(12.),hash(105.),hash(156.)), 
                          vec3(hash(32.),hash(123.),hash(25.)),         
                          vec3(hash(10.),hash(15.),hash(27.)));               
                        
}

col = col * linear;
col += 5. * spe * vec3(hash(146.),hash(925.),hash(547.));

col = mix(col,vec3(1.),1. - exp(-.0001 * d.x * d.x * d.x));

} 

return col;
}

void main() {
 
vec3 color = vec3(0.);

vec3 cam_tar = vec3(0.);
vec3 cam_pos = cameraPosition;

vec2 uvu = -1. + 2. * uVu.xy; 
uvu.x *= res.x/res.y; 

vec3 dir = rayCamDir(uvu,cam_pos,cam_tar,2.); 
color = render(cam_pos,dir);  

out_FragColor = vec4(color,1.0);

}
`;

let renderer,canvas,context;

let uniforms;

let scene,material,mesh;

let plane,w,h; 

let cam,controls,target;
let sphere,sphere_mat;

let r = new Math.seedrandom();
let s = Math.abs(r.int32());

let cast = {

    steps : 350,
    eps : 0.00001,
    dmin : 0.,
    dmax : 500.

};

let gui = new dat.GUI();

let castfolder = gui.addFolder('cast');

castfolder.add(cast,'steps',500).onChange(updateUniforms);
castfolder.add(cast,'eps',0.00001).onChange(updateUniforms); 
castfolder.add(cast,'dmin',0.,10.).onChange(updateUniforms);
castfolder.add(cast,'dmax',0.,2000.).onChange(updateUniforms);

init();
render();

function init() {

    canvas = $('#canvas')[0];
    context = canvas.getContext('webgl2');
    
    w = window.innerWidth;
    h = window.innerHeight;

    canvas.width = w;
    canvas.height = h;

    scene = new THREE.Scene();
    plane = new THREE.PlaneBufferGeometry(2,2);

    renderer = new THREE.WebGLRenderer({
    
        canvas : canvas,
        context : context

    });

    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w,h);
    
    cam = new THREE.PerspectiveCamera(0.,w/h,0.,1.);
    cam.position.set(1.,5.,1.15);
    
    sphere = new THREE.SphereBufferGeometry();
    sphere_mat = new THREE.Material();
    target = new THREE.Mesh(sphere,sphere_mat);
    target.position.set(0.,0.,0.);
    target.add(cam);
    
    cam.lookAt(target);

    material = new THREE.ShaderMaterial({

       uniforms : {
    
           res        : new THREE.Uniform(new THREE.Vector2(w,h)),    
           time          : { value : 1. },
           seed          : { value : s },
           steps         : { value : cast.steps },
           eps           : { value : cast.eps },
           dmin          : { value : cast.dmin },
           dmax          : { value : cast.dmax }

       },

       vertexShader   : vert,
       fragmentShader : frag

    });

    mesh = new THREE.Mesh(plane,material);
    scene.add(mesh);

}

function updateUniforms() {

    material.uniforms.steps.value = cast.steps;
    material.uniforms.eps.value = cast.eps;
    material.uniforms.dmin.value = cast.dmin;
    material.uniforms.dmax.value = cast.dmax;

}
    function render() {

    updateUniforms();
 
    material.uniforms.res.value = new THREE.Vector2(w,h);
    material.uniforms.time.value = performance.now();

    renderer.render(scene,cam);
    requestAnimationFrame(render);

}


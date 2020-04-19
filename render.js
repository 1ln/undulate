let w,h;

let canvas,context;

let renderer;
let render;
let uniforms;

let reset;

let nhash,hash;  

let mouse_pressed,mouse_held,mouse;

let cam,scene,geometry,mesh,mat;

let cam_target;

let box_material;
let box_geometry;
let box;

let angle;

let light,light2,light3;
let quatern,quatern2,quatern3;

function init() {

    canvas  = $('#canvas')[0];
    context = canvas.getContext('webgl2',{ antialias:false });

    w = window.innerWidth;
    h = window.innerHeight;

    renderer = new THREE.WebGLRenderer({canvas:canvas,context:context});

    cam = new THREE.PerspectiveCamera(45.,w/h,0.0,1000.0);

    angle = Math.PI / 2. ;

    box_geometry = new THREE.BoxBufferGeometry(1,1,100);
    box_material = new THREE.MeshBasicMaterial({color: '#FF0000'});
    box = new THREE.Mesh(box_geometry,box_material);
    box.position.set(0,0,0);

    nhash = new Math.seedrandom();
    hash = nhash();

    mouse = new THREE.Vector2(0.0); 
    mouse_pressed = 0;
    mouse_held = 0;

    cam.position.set(0,0,5); 
    cam_target  = new THREE.Vector3(0.0);

    light = new THREE.PointLight(0x11,1,100);
    light.position.set(hash()*100.,hash()*100.,hash()*100.);

    light2 = new THREE.PointLight(0x11,1,100);
    light2.position.set(hash()*100.,hash()*100.,hash()*100.);

    light3 = new THREE.PointLight(0x11,1,100);
    light3.position.set(hash()*100.,hash()*100.,hash()*100.);

    quatern = new THREE.Quaternion();
    quatern2 = new THREE.Quaternion();
    quatern3 = new THREE.Quaternion();

    scene = new THREE.Scene();

    geometry = new THREE.PlaneBufferGeometry(2,2);

    uniforms = {

        "u_time"                : { value : 1.0 },
        "u_resolution"          : new THREE.Uniform(new THREE.Vector2(w,h)),
        "u_mouse"               : new THREE.Uniform(new THREE.Vector2()),
        "u_mouse_pressed"       : { value : mouse_pressed },
        "u_light_pos"           : new THREE.Uniform(new THREE.Vector3(light.position)),
        "u_light2_pos"          : new THREE.Uniform(new THREE.Vector3(light2.position)),
        "u_light3_pos"          : new THREE.Uniform(new THREE.Vector3(light3.position)),        
        "u_cam_target"          : new THREE.Uniform(new THREE.Vector3(cam_target)),
        "u_hash"                : { value: hash }

    };   

}

init();

ShaderLoader("render.vert","render.frag",

    function(vertex,fragment) {

        material = new THREE.ShaderMaterial({

            uniforms : uniforms,
            vertexShader : vertex,
            fragmentShader : fragment

        });

        mesh = new THREE.Mesh(geometry,material);

        scene.add(mesh);        

        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(w,h);

        render = function(timestamp) {

        requestAnimationFrame(render);
    
        uniforms["u_time"                ].value = performance.now();
        uniforms["u_mouse"               ].value = mouse;
        uniforms["u_mouse_pressed"       ].value = mouse_pressed;
        uniforms["u_light_pos"           ].value = light.position;
        uniforms["u_light2_pos"          ].value = light2.position;
        uniforms["u_light3_pos"          ].value = light3.position;
        uniforms["u_cam_target"          ].value = cam_target;
        uniforms["u_hash"                ].value = hash;      

        quatern.setFromAxisAngle(new THREE.Vector3(0,1,0),angle);
        quatern2.setFromAxisAngle(new THREE.Vector3(0,1,0),angle);
        quatern3.setFromAxisAngle(new THREE.Vector3(0,1,0),angle);

        renderer.render(scene,cam);

        } 
       
    render();

    }
) 

$('#canvas').mousedown(function() { 
 
    mouse_pressed = true;
   
    reset = setTimeout(function() {
    mouse_held = true; 


    },5000);


});

$('#canvas').mouseup(function() {
    
    mouse_pressed = false;    
    mouse_held = false;
    

    if(reset) {
        clearTimeout(reset);
    };

});        

window.addEventListener('mousemove',onMouseMove,false);

function onMouseMove(event) {
    mouse.x = (event.clientX / w) * 2.0 - 1.0; 
    mouse.y = -(event.clientY / h) * 2.0 + 1.0;
}

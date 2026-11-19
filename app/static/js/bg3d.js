/*
 * Harvest Guard — realistic 3D crop-field background, raw WebGL2 (no libraries, no CDN).
 *
 *   • physically-flavoured sky: gradient, sun disc + glow, fBm clouds
 *   • rolling terrain with ploughed rows, analytic normals, aerial-perspective fog
 *   • ~50k instanced, wind-animated crop blades with back-lit translucency
 *   • floating pollen / dust particles
 *   • mouse parallax + scroll-driven camera lift, adaptive quality, reduced-motion support
 *
 * Usage:  HGBackground.start(document.getElementById('bg'))
 */
(function () {
  'use strict';

  // ------------------------------------------------------------------ small mat4 library (column-major)
  var M = {
    persp: function (fovy, asp, n, f) { var t = 1 / Math.tan(fovy / 2), o = new Float32Array(16); o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; },
    look: function (e, c, u) {
      var z = norm(sub(e, c)), x = norm(cross(u, z)), y = cross(z, x), o = new Float32Array(16);
      o[0] = x[0]; o[1] = y[0]; o[2] = z[0]; o[4] = x[1]; o[5] = y[1]; o[6] = z[1]; o[8] = x[2]; o[9] = y[2]; o[10] = z[2];
      o[12] = -dot(x, e); o[13] = -dot(y, e); o[14] = -dot(z, e); o[15] = 1; return o;
    },
    mul: function (a, b) { var o = new Float32Array(16); for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) { var s = 0; for (var k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; },
    inv: function (m) {
      var inv = new Float32Array(16), d;
      inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
      inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
      inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
      inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
      inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
      inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
      inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
      inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
      inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
      inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
      inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
      inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
      inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
      inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
      inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
      inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
      d = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12]; d = 1 / d;
      for (var i = 0; i < 16; i++) inv[i] *= d; return inv;
    }
  };
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function norm(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
  function smooth(a, b, x) { var t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

  // Terrain height — MUST match terrainH() in GLSL below.
  function terrainH(x, z) {
    var h = 0.9 * Math.sin(x * 0.045 + 1.3) * Math.cos(z * 0.037) + 0.5 * Math.sin(x * 0.11 + z * 0.07 + 2.0) + 0.12 * Math.sin(x * 0.4 + z * 0.33);
    var far = smooth(80, 220, -z);
    h += far * (14 * (0.5 + 0.5 * Math.sin(x * 0.021 + 1.0)) + 3 * (0.5 + 0.5 * Math.sin(x * 0.06 + z * 0.02)));
    return h;
  }

  // ------------------------------------------------------------------ shaders
  var COMMON = [
    '#version 300 es',
    'precision highp float;',
    'uniform mat4 uVP; uniform vec3 uCam; uniform float uTime; uniform vec3 uSun;',
    'const vec3 FOG = vec3(0.86, 0.74, 0.58);',
    'float terrainH(vec2 p){',
    '  float h = 0.9*sin(p.x*0.045+1.3)*cos(p.y*0.037) + 0.5*sin(p.x*0.11+p.y*0.07+2.0) + 0.12*sin(p.x*0.4+p.y*0.33);',
    '  float far = smoothstep(80.0,220.0,-p.y);',
    '  h += far*(14.0*(0.5+0.5*sin(p.x*0.021+1.0)) + 3.0*(0.5+0.5*sin(p.x*0.06+p.y*0.02)));',
    '  return h; }',
    'vec3 applyFog(vec3 c, float dist, float y){',
    '  float f = 1.0 - exp(-dist*0.0105*(1.0+max(0.0,3.0-y)*0.12));',
    '  return mix(c, FOG, clamp(f,0.0,1.0)); }',
    'float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }',
    'vec3 tonemap(vec3 c){ c = c*0.72; c = (c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14); return pow(clamp(c,0.0,1.0), vec3(1.0/2.2)); }'
  ].join('\n');

  var SKY_VS = '#version 300 es\nout vec2 vNdc; void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); vNdc = p*2.0-1.0; gl_Position = vec4(vNdc, 1.0, 1.0); }';
  var SKY_FS = COMMON + '\n' + [
    'uniform mat4 uInvVP; in vec2 vNdc; out vec4 o;',
    'float noise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);',
    '  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }',
    'float fbm(vec2 p){ float a=0.5, s=0.0; for(int i=0;i<5;i++){ s+=a*noise(p); p=p*2.03+vec2(7.1,3.7); a*=0.5; } return s; }',
    'void main(){',
    '  vec4 w = uInvVP*vec4(vNdc,1.0,1.0); vec3 d = normalize(w.xyz/w.w - uCam);',
    '  float t = clamp(d.y,0.0,1.0);',
    '  vec3 zenith = vec3(0.10,0.27,0.58), mid = vec3(0.40,0.58,0.80), hor = vec3(0.98,0.76,0.52);',
    '  vec3 col = mix(hor, mid, smoothstep(0.0,0.22,t)); col = mix(col, zenith, smoothstep(0.18,0.85,t));',
    '  float s = max(dot(d, uSun), 0.0);',
    '  col += vec3(1.0,0.72,0.40)*(pow(s,6.0)*0.22 + pow(s,40.0)*0.45) + vec3(1.0,0.95,0.82)*pow(s,1400.0)*10.0;',
    '  // clouds: project ray onto a plane high above the camera',
    '  if (d.y > 0.015) {',
    '    vec2 p = d.xz/(d.y+0.12)*1.35 + vec2(uTime*0.018, uTime*0.006);',
    '    float n = fbm(p*0.9), n2 = fbm(p*2.1+3.0);',
    '    float cov = smoothstep(0.58,0.86,n*0.8+n2*0.35);',
    '    float lit = clamp(0.55 + 0.45*(fbm(p*0.9 + uSun.xz*0.25) - n)*6.0, 0.0, 1.0);',
    '    vec3 cc = mix(vec3(0.58,0.56,0.62), vec3(1.0,0.93,0.82), lit);',
    '    cc += vec3(1.0,0.6,0.3)*pow(s,8.0)*0.5;',
    '    float fade = smoothstep(0.015,0.2,d.y);',
    '    col = mix(col, cc, cov*0.92*fade);',
    '  }',
    '  col = mix(col, FOG, smoothstep(0.02,-0.12,d.y)); // below horizon',
    '  o = vec4(tonemap(col),1.0); }'
  ].join('\n');

  var TER_VS = COMMON + '\n' + [
    'in vec2 aXZ; out vec3 vP; out vec3 vN;',
    'void main(){',
    '  float y = terrainH(aXZ); vec3 p = vec3(aXZ.x, y, aXZ.y);',
    '  float e = 0.6; vec3 n = normalize(vec3(terrainH(aXZ-vec2(e,0.0))-terrainH(aXZ+vec2(e,0.0)), 2.0*e, terrainH(aXZ-vec2(0.0,e))-terrainH(aXZ+vec2(0.0,e))));',
    '  vP = p; vN = n; gl_Position = uVP*vec4(p,1.0); }'
  ].join('\n');
  var TER_FS = COMMON + '\n' + [
    'in vec3 vP; in vec3 vN; out vec4 o;',
    'void main(){',
    '  vec3 n = normalize(vN);',
    '  float row = 0.5+0.5*cos(vP.x*6.2831/0.7);',
    '  float jitter = 0.5*hash(floor(vP.xz*vec2(5.0,7.0))) + 0.5*hash(floor(vP.xz*vec2(1.3,1.9)));',
    '  vec3 soil = mix(vec3(0.20,0.12,0.07), vec3(0.34,0.22,0.12), jitter*0.6 + (1.0-row)*0.5);',
    '  vec3 green = vec3(0.10,0.20,0.05) + 0.05*jitter;',
    '  float near = smoothstep(75.0, 20.0, length(vP.xz-uCam.xz));',
    '  vec3 alb = mix(mix(green, vec3(0.22,0.30,0.10), 0.5*jitter), soil, near*(1.0-row*0.55));',
    '  float far = smoothstep(80.0,200.0,-vP.z); alb = mix(alb, vec3(0.20,0.26,0.16), far*0.7);',
    '  float dl = max(dot(n,uSun),0.0);',
    '  vec3 lit = alb*(vec3(1.0,0.78,0.52)*dl*1.25 + vec3(0.30,0.38,0.52)*(0.45+0.25*n.y));',
    '  float dist = length(vP - uCam);',
    '  o = vec4(tonemap(applyFog(lit, dist, vP.y)),1.0); }'
  ].join('\n');

  var BLADE_VS = COMMON + '\n' + [
    'in vec2 aV;            // x = t along blade (0..1), y = side (-1|1)',
    'in vec4 iA;            // base.xyz, height',
    'in vec4 iB;            // width, phase, yaw, hue',
    'out vec3 vCol; out vec3 vP; out float vT; out vec3 vN;',
    'uniform float uWind;',
    'void main(){',
    '  float t = aV.x, side = aV.y; float h = iA.w, w = iB.x, ph = iB.y, yaw = iB.z, hue = iB.w;',
    '  vec3 right = vec3(cos(yaw), 0.0, sin(yaw));',
    '  vec3 base = iA.xyz;',
    '  float wv = sin(uTime*1.7 + base.x*0.33 + base.z*0.29 + ph)*0.16 + sin(uTime*0.55 + base.z*0.11 - base.x*0.05)*0.26 + sin(uTime*3.1 + ph*3.0)*0.03;',
    '  vec3 lean = vec3(wv*uWind + (hue-0.5)*0.35, 0.0, wv*0.55*uWind + sin(ph)*0.2);',
    '  float bend = t*t;',
    '  vec3 pos = base + vec3(0.0, t*h*(1.0-0.18*bend*length(lean)), 0.0) + lean*h*bend + right*side*w*(1.0-t*0.88);',
    '  vec3 fwd = vec3(-sin(yaw),0.0,cos(yaw)); vec3 n = normalize(fwd + vec3(0.0, 0.5*t, 0.0) + lean*bend);',
    '  vec3 baseCol = vec3(0.045,0.11,0.025), tipCol = mix(vec3(0.17,0.36,0.07), vec3(0.45,0.50,0.12), smoothstep(0.7,1.0,hue));',
    '  if (hue > 0.965) tipCol = vec3(0.50,0.36,0.12);',
    '  vCol = mix(baseCol, tipCol, smoothstep(0.0,1.0,t)*(0.65+0.5*hue)); vP = pos; vT = t; vN = n;',
    '  gl_Position = uVP*vec4(pos,1.0); }'
  ].join('\n');
  var BLADE_FS = COMMON + '\n' + [
    'in vec3 vCol; in vec3 vP; in float vT; in vec3 vN; out vec4 o;',
    'void main(){',
    '  vec3 V = normalize(uCam - vP); vec3 n = normalize(vN); if (dot(n,V) < 0.0) n = -n;',
    '  float dl = max(dot(n,uSun),0.0)*0.6 + 0.4;',
    '  float back = pow(max(dot(-V,uSun),0.0), 2.5);          // sun shining through the blade toward the camera',
    '  vec3 sunC = vec3(1.0,0.78,0.5);',
    '  vec3 c = vCol*(sunC*dl*1.15 + vec3(0.28,0.36,0.5)*0.55) + vCol*vec3(1.0,0.9,0.4)*back*vT*1.5;',
    '  c *= 0.42 + 0.58*smoothstep(0.0,0.5,vT);                // contact shadow at the base',
    '  float dist = length(vP - uCam);',
    '  o = vec4(tonemap(applyFog(c, dist, vP.y)),1.0); }'
  ].join('\n');

  var PART_VS = COMMON + '\n' + [
    'in vec4 aP; // x,y,z seed ; w = size',
    'out float vA;',
    'void main(){',
    '  vec3 p = aP.xyz; float s = aP.w;',
    '  p.x += sin(uTime*0.3 + s*40.0)*1.5 + uTime*0.35; p.y += sin(uTime*0.5 + s*20.0)*0.6; p.z += cos(uTime*0.25 + s*30.0)*1.2;',
    '  p.x = mod(p.x + 30.0, 60.0) - 30.0;',
    '  vec4 c = uVP*vec4(p,1.0); gl_Position = c;',
    '  gl_PointSize = clamp((1.5 + s*5.0) * 28.0 / max(c.w, 0.5), 1.0, 12.0);',
    '  vA = clamp(1.0 - c.w/40.0, 0.0, 1.0)*(0.25+0.5*s); }'
  ].join('\n');
  var PART_FS = '#version 300 es\nprecision mediump float; in float vA; out vec4 o; void main(){ vec2 d = gl_PointCoord-0.5; float r = dot(d,d)*4.0; if (r>1.0) discard; float a = (1.0-r)*(1.0-r)*vA; o = vec4(vec3(1.0,0.86,0.58)*a*0.8, a*0.8); }';

  // ------------------------------------------------------------------ helpers
  function compile(gl, type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('Shader error: ' + gl.getShaderInfoLog(s));
    return s;
  }
  function program(gl, vs, fs) {
    var p = gl.createProgram(); gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Link error: ' + gl.getProgramInfoLog(p));
    return p;
  }
  function rand(seed) { var a = seed >>> 0; return function () { a = (a + 0x6d2b79f5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  // ------------------------------------------------------------------ main
  function start(canvas, opts) {
    opts = opts || {};
    var gl = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'high-performance' });
    if (!gl) { document.documentElement.classList.add('no-webgl'); return null; }
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var P;
    try {
      P = { sky: program(gl, SKY_VS, SKY_FS), ter: program(gl, TER_VS, TER_FS), blade: program(gl, BLADE_VS, BLADE_FS), part: program(gl, PART_VS, PART_FS) };
    } catch (e) { console.warn('[bg3d]', e.message); document.documentElement.classList.add('no-webgl'); return null; }

    var sun = norm([-0.42, 0.30, -0.86]);

    // terrain grid
    var TX = 150, TZ = 150, X0 = -170, X1 = 170, Z0 = -260, Z1 = 40;
    var tv = new Float32Array((TX + 1) * (TZ + 1) * 2), ti = new Uint32Array(TX * TZ * 6), k = 0, q = 0, i, j;
    for (j = 0; j <= TZ; j++) for (i = 0; i <= TX; i++) {
      var u = i / TX, v = j / TZ;
      // denser near the camera (z close to Z1)
      tv[k++] = X0 + (X1 - X0) * u; tv[k++] = Z1 - (Z1 - Z0) * Math.pow(v, 1.7);
    }
    for (j = 0; j < TZ; j++) for (i = 0; i < TX; i++) { var a = j * (TX + 1) + i, b = a + 1, c = a + TX + 1, d = c + 1; ti[q++] = a; ti[q++] = c; ti[q++] = b; ti[q++] = b; ti[q++] = c; ti[q++] = d; }
    var terVAO = gl.createVertexArray(); gl.bindVertexArray(terVAO);
    var tb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, tb); gl.bufferData(gl.ARRAY_BUFFER, tv, gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(P.ter, 'aXZ'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    var teb = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, teb); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ti, gl.STATIC_DRAW);

    // blade geometry: 5 rings x 2 verts as a triangle strip
    var SEG = 5, bv = new Float32Array(SEG * 2 * 2);
    for (i = 0; i < SEG; i++) { var t = i / (SEG - 1); bv.set([t, -1, t, 1], i * 4); }
    var maxBlades = opts.blades || 52000, blades = new Float32Array(maxBlades * 8), r = rand(7);
    var rowSpacing = 0.7;
    for (i = 0; i < maxBlades; i++) {
      var zz = 7 - 78 * Math.pow(r(), 1.7);
      var row = Math.round((r() * 2 - 1) * 34), xx = row * rowSpacing + (r() - 0.5) * 0.2;
      var yy = terrainH(xx, zz) - 0.02, hh = 0.55 + r() * 0.6;
      blades.set([xx, yy, zz, hh, 0.022 + r() * 0.022, r() * 6.283, r() * 6.283, r()], i * 8);
    }
    var bladeVAO = gl.createVertexArray(); gl.bindVertexArray(bladeVAO);
    var bb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bb); gl.bufferData(gl.ARRAY_BUFFER, bv, gl.STATIC_DRAW);
    loc = gl.getAttribLocation(P.blade, 'aV'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    var ib = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ib); gl.bufferData(gl.ARRAY_BUFFER, blades, gl.STATIC_DRAW);
    loc = gl.getAttribLocation(P.blade, 'iA'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 32, 0); gl.vertexAttribDivisor(loc, 1);
    loc = gl.getAttribLocation(P.blade, 'iB'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 32, 16); gl.vertexAttribDivisor(loc, 1);

    // particles
    var NP = 380, pd = new Float32Array(NP * 4), pr = rand(99);
    for (i = 0; i < NP; i++) pd.set([(pr() - 0.5) * 60, 0.3 + pr() * 6, 6 - pr() * 45, pr()], i * 4);
    var partVAO = gl.createVertexArray(); gl.bindVertexArray(partVAO);
    var pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, pd, gl.STATIC_DRAW);
    loc = gl.getAttribLocation(P.part, 'aP'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
    var skyVAO = gl.createVertexArray();

    function U(p, n) { return gl.getUniformLocation(p, n); }
    var uni = {};
    ['sky', 'ter', 'blade', 'part'].forEach(function (n) { uni[n] = { vp: U(P[n], 'uVP'), cam: U(P[n], 'uCam'), time: U(P[n], 'uTime'), sun: U(P[n], 'uSun'), inv: U(P[n], 'uInvVP'), wind: U(P[n], 'uWind') }; });

    // state
    var mouse = { x: 0, y: 0, tx: 0, ty: 0 }, scroll = 0, scrollT = 0, active = blades.length / 8, scale = Math.min(window.devicePixelRatio || 1, 1.5), frames = 0, acc = 0, last = performance.now(), t0 = last, raf = 0, running = true;
    window.addEventListener('pointermove', function (e) { mouse.tx = e.clientX / innerWidth * 2 - 1; mouse.ty = e.clientY / innerHeight * 2 - 1; }, { passive: true });
    window.addEventListener('scroll', function () { var m = document.documentElement.scrollHeight - innerHeight; scrollT = m > 0 ? Math.min(1, scrollY / Math.min(m, 900)) : 0; }, { passive: true });

    function resize() {
      var w = Math.max(1, Math.floor(canvas.clientWidth * scale)), h = Math.max(1, Math.floor(canvas.clientHeight * scale));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    }
    function frame(now) {
      raf = 0;
      var dt = Math.min(0.1, (now - last) / 1000); last = now;
      var time = reduce ? 4.0 : (now - t0) / 1000;
      mouse.x += (mouse.tx - mouse.x) * 0.05; mouse.y += (mouse.ty - mouse.y) * 0.05; scroll += (scrollT - scroll) * 0.06;
      resize();
      gl.viewport(0, 0, canvas.width, canvas.height);
      var eye = [mouse.x * 0.8, terrainH(0, 8) + 2.3 + scroll * 4.5, 8];
      var tgt = [mouse.x * 9, eye[1] - 1.9 - scroll * 2.8 - mouse.y * 0.9, -22];
      var proj = M.persp(1.0, canvas.width / canvas.height, 0.1, 500), view = M.look(eye, tgt, [0, 1, 0]), vp = M.mul(proj, view), inv = M.inv(vp);

      gl.clearColor(0.86, 0.74, 0.58, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      function setU(n, withInv) {
        var u = uni[n]; gl.uniformMatrix4fv(u.vp, false, vp); gl.uniform3fv(u.cam, eye); gl.uniform1f(u.time, time); gl.uniform3fv(u.sun, sun);
        if (withInv && u.inv) gl.uniformMatrix4fv(u.inv, false, inv);
      }
      gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.useProgram(P.sky); setU('sky', true); gl.bindVertexArray(skyVAO); gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
      gl.useProgram(P.ter); setU('ter'); gl.bindVertexArray(terVAO); gl.drawElements(gl.TRIANGLES, ti.length, gl.UNSIGNED_INT, 0);
      gl.useProgram(P.blade); setU('blade'); gl.uniform1f(uni.blade.wind, reduce ? 0.3 : 1.0); gl.bindVertexArray(bladeVAO); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, SEG * 2, active | 0);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
      gl.useProgram(P.part); setU('part'); gl.bindVertexArray(partVAO); gl.drawArrays(gl.POINTS, 0, NP);
      gl.depthMask(true); gl.disable(gl.BLEND);

      // adaptive quality: if we are consistently slow, drop blades, then resolution
      acc += dt; frames++;
      if (frames === 40) {
        var avg = acc / frames; acc = 0; frames = 0;
        if (avg > 0.034 && !opts.fixed) { if (active > 14000) active *= 0.65; else if (scale > 0.6) scale *= 0.85; }
      }
      if (!reduce && running) raf = requestAnimationFrame(frame);
    }
    function kick() { if (!raf) raf = requestAnimationFrame(frame); }
    document.addEventListener('visibilitychange', function () { running = !document.hidden; if (running) { last = performance.now(); kick(); } });
    window.addEventListener('resize', function () { if (reduce) { cancelAnimationFrame(raf); raf = 0; kick(); } });
    canvas.addEventListener('webglcontextlost', function (e) { e.preventDefault(); running = false; });
    canvas.addEventListener('webglcontextrestored', function () { location.reload(); });
    kick();
    return { gl: gl, setQuality: function (n) { active = Math.min(maxBlades, n); } };
  }

  window.HGBackground = { start: start };
})();

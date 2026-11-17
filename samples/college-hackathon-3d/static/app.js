/* HackVerse 2026 frontend: 3D scene + API wiring. */

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- Three.js hero scene ---------- */
(function initScene() {
  if (typeof THREE === "undefined") return;
  const canvas = document.getElementById("scene");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    65,
    window.innerWidth / window.innerHeight,
    0.1,
    100
  );
  camera.position.z = 16;

  const palette = [0x7c5cff, 0x00d4ff, 0xff4fa3];

  // Particle starfield.
  const count = 1400;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const color = new THREE.Color();
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 46;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 30;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 26;
    color.setHex(palette[Math.floor(Math.random() * palette.length)]);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  const particlesGeo = new THREE.BufferGeometry();
  particlesGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  particlesGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const particles = new THREE.Points(
    particlesGeo,
    new THREE.PointsMaterial({ size: 0.075, vertexColors: true, transparent: true, opacity: 0.85 })
  );
  scene.add(particles);

  // Central wireframe icosahedron.
  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(3.1, 1),
    new THREE.MeshBasicMaterial({
      color: 0x7c5cff,
      wireframe: true,
      transparent: true,
      opacity: 0.45
    })
  );
  core.position.set(6.2, 0.5, -3);
  scene.add(core);

  // Orbiting shards.
  const shards = [];
  for (let i = 0; i < 5; i++) {
    const shard = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.55 + Math.random() * 0.4, 0),
      new THREE.MeshBasicMaterial({
        color: palette[i % palette.length],
        wireframe: true,
        transparent: true,
        opacity: 0.7
      })
    );
    shard.userData.radius = 4.5 + i * 0.6;
    shard.userData.speed = 0.12 + i * 0.04;
    shard.userData.phase = i * 1.3;
    shard.position.set(6.2, 0.5, -3);
    scene.add(shard);
    shards.push(shard);
  }

  let mx = 0;
  let my = 0;
  window.addEventListener("pointermove", (event) => {
    mx = (event.clientX / window.innerWidth - 0.5) * 2;
    my = (event.clientY / window.innerHeight - 0.5) * 2;
  });
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const clock = new THREE.Clock();
  function frame() {
    const t = clock.getElapsedTime();
    if (!reduceMotion) {
      particles.rotation.y = t * 0.015;
      particles.rotation.x = mx * 0.05;
      core.rotation.x = t * 0.12 + my * 0.15;
      core.rotation.y = t * 0.18 + mx * 0.15;
      shards.forEach((shard) => {
        const { radius, speed, phase } = shard.userData;
        shard.position.x = 6.2 + Math.cos(t * speed + phase) * radius;
        shard.position.y = 0.5 + Math.sin(t * speed * 1.3 + phase) * radius * 0.5;
        shard.position.z = -3 + Math.sin(t * speed + phase) * radius * 0.6;
        shard.rotation.x = t * 0.5;
        shard.rotation.y = t * 0.3;
      });
      camera.position.x += (mx * 0.8 - camera.position.x) * 0.03;
      camera.position.y += (-my * 0.5 - camera.position.y) * 0.03;
      camera.lookAt(0, 0, 0);
    }
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  frame();
})();

/* ---------- Data loading ---------- */
async function getJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

async function loadTracks() {
  const list = document.getElementById("track-list");
  const select = document.getElementById("track-select");
  try {
    const tracks = await getJSON("/api/tracks");
    list.innerHTML = tracks
      .map(
        (track) => `
        <article class="card">
          <span class="track-color" style="background:${track.color}"></span>
          <h3>${track.name}</h3>
          <p>Prizes and judges focused on ${track.name.toLowerCase()}.</p>
        </article>`
      )
      .join("");
    tracks.forEach((track) => {
      const option = document.createElement("option");
      option.value = track.id;
      option.textContent = track.name;
      select.appendChild(option);
    });
  } catch (error) {
    list.innerHTML = `<p class="muted">Could not load tracks (${error.message}).</p>`;
  }
}

async function loadSchedule() {
  const list = document.getElementById("schedule-list");
  try {
    const items = await getJSON("/api/schedule");
    list.innerHTML = items
      .map(
        (item) => `
        <div class="timeline-item">
          <div class="timeline-time">${item.time} · DAY ${item.day}</div>
          <div class="timeline-title">${item.title}</div>
        </div>`
      )
      .join("");
  } catch (error) {
    list.innerHTML = `<p class="muted">Could not load schedule (${error.message}).</p>`;
  }
}

async function loadStats() {
  try {
    const stats = await getJSON("/api/stats");
    document.querySelector('[data-stat="teams"]').textContent = stats.teams;
    document.querySelector('[data-stat="tracks"]').textContent = stats.tracks;
    document.querySelector('[data-stat="members"]').textContent = stats.members;
  } catch (_error) {
    // /api/stats is not implemented yet; placeholders stay as dashes.
  }
}

/* ---------- Registration ---------- */
document.getElementById("register-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const status = document.getElementById("form-status");
  const payload = {
    name: form.name.value.trim(),
    track: form.track.value,
    members: Number(form.members.value)
  };
  status.className = "form-status";
  status.textContent = "Registering…";
  try {
    const response = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Registration failed");
    status.className = "form-status ok";
    status.textContent = `✓ ${body.name} is registered!`;
    form.reset();
    form.members.value = 3;
    loadStats();
  } catch (error) {
    status.className = "form-status err";
    status.textContent = error.message;
  }
});

loadTracks();
loadSchedule();
loadStats();

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      // Non-JSON error response; keep the generic message.
    }
    throw new Error(detail);
  }

  if (response.status === 204) return null;
  return response.json();
}

export const projectsApi = {
  list: () => request('/projects/'),
  create: (name) => request('/projects/', { method: 'POST', body: JSON.stringify({ name }) }),
  update: (id, name) =>
    request(`/projects/${id}`, { method: 'PUT', body: JSON.stringify({ name }) }),
  remove: (id) => request(`/projects/${id}`, { method: 'DELETE' }),
};

export const tasksApi = {
  list: ({ projectId, status } = {}) => {
    const params = new URLSearchParams();
    if (projectId) params.set('project_id', projectId);
    if (status && status !== 'all') params.set('status', status);
    const query = params.toString();
    return request(`/tasks/${query ? `?${query}` : ''}`);
  },
  create: (task) => request('/tasks/', { method: 'POST', body: JSON.stringify(task) }),
  update: (id, task) => request(`/tasks/${id}`, { method: 'PUT', body: JSON.stringify(task) }),
  remove: (id) => request(`/tasks/${id}`, { method: 'DELETE' }),
};

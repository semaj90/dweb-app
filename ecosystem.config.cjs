module.exports = {
  apps: [
    {
      name: 'yorha-web',
      script: 'npm',
      args: 'run dev',
      cwd: './',
      env: {
        NODE_ENV: 'development',
        PORT: 5173
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      restart_delay: 3000,
      max_restarts: 5,
      min_uptime: '10s'
    }
  ]
}
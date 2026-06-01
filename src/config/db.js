const { Pool } = require("pg");
require("dotenv").config();

const pool = new Pool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,

  ssl: {
    rejectUnauthorized: false,
  },
});

pool.connect((err, client, release) => {
  if (err) {
    console.log("Database connection error:");
    console.log(err.message);
  } else {
    console.log("Connect PostgreSQL success");
  }

  if (client) {
    release();
  }
});

module.exports = pool;

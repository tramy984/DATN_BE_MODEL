const express = require("express");
const dotenv = require("dotenv");
const recommendationRoutes = require("./routes/recommendation.routes");

dotenv.config();

const app = express();

app.use(express.json());

app.get("/recommendation", (req, res) => {
  res.json({
    message: "Welcome to the AI model API project",
  });
});

app.use("/api", recommendationRoutes);

module.exports = app;
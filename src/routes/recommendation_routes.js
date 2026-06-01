// const express = require("express");
// const axios = require("axios");
// const pool = require("../config/db");

// const router = express.Router();

// router.post("/recommendation/jobs", async (req, res) => {
//   try {
//     const { cv_text, top_k = 10 } = req.body;

//     if (!cv_text) {
//       return res.status(400).json({
//         success: false,
//         message: "Thiếu cv_text",
//       });
//     }

//     // 1. Gọi Python AI service
//     const aiResponse = await axios.post(
//       `${process.env.AI_SERVICE_URL}/recommend`,
//       {
//         cv_text,
//         top_k,
//       }
//     );

//     const recommendations = aiResponse.data.data;

//     if (!recommendations || recommendations.length === 0) {
//       return res.json({
//         success: true,
//         data: [],
//       });
//     }

//     const jobIds = recommendations.map((item) => item.job_id);

//     // 2. Query thông tin job từ PostgreSQL
//     const query = `
//       SELECT 
//         j.id,
//         j.name,
//         j.description,
//         j.requirement,
//         j.salary_min,
//         j.salary_max,
//         j.salary_type,
//         j.location,
//         c.name AS company_name
//       FROM jobs j
//       LEFT JOIN company c ON c.id = j.company_id
//       WHERE j.id = ANY($1::int[])
//     `;

//     const { rows } = await pool.query(query, [jobIds]);

//     // 3. Map score vào job
//     const scoreMap = new Map(
//       recommendations.map((item) => [
//         Number(item.job_id),
//         item.score,
//       ])
//     );

//     const jobMap = new Map(
//       rows.map((job) => [
//         Number(job.id),
//         job,
//       ])
//     );

//     const result = jobIds
//       .map((jobId) => {
//         const job = jobMap.get(Number(jobId));

//         if (!job) return null;

//         return {
//           ...job,
//           ai_score: scoreMap.get(Number(jobId)),
//         };
//       })
//       .filter(Boolean);

//     return res.json({
//       success: true,
//       data: result,
//     });

//   } catch (error) {
//     console.error("Recommendation error:", error.message);

//     return res.status(500).json({
//       success: false,
//       message: "Lỗi recommendation",
//     });
//   }
// });

// module.exports = router;
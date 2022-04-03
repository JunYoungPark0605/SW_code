const express = require('express');
const router = express.Router();
const { Star } = require("../models/Star");
const { auth } = require("../middleware/auth");

router.post("/getStar", (req, res) => {
    let variable = { productId: req.body.productId, userId: req.body.userId };
    Star.find(variable).exec((err, stars) => {
        if (err) return res.status(400).send(err)
        res.status(200).json({ success: true, stars })
    })
})

router.post("/upStar", (req, res) => {
    let variable = { 
        productId: req.body.productId, 
        userId: req.body.userId,
        star: req.body.star
    };
    let save = req.body.save;
    
    if (save) {
        const star = new Star(variable);
        star.save((err, starSaveResult) => {
            if (err) return res.status(400).send(err);
            res.status(200).json({ success: true, starSaveResult})
        })
    } else {
        Star.findOneAndUpdate(
            { productId: req.body.productId, userId: req.body.userId },
            { $set: { star: req.body.star }},
            { new: true },
            (err, starUpdateResult) => {
                res.status(200).json({ success: true, starUpdateResult })
            }
        )
    }
})

module.exports = router;
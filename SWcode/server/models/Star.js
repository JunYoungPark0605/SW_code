const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const starSchema = mongoose.Schema({
   userId: {
       type: Schema.Types.ObjectId,
       ref: 'User'
   },
   productId: {
       type: Schema.Types.ObjectId,
       ref: 'Product'
   },
   star: {
        type: Number,
        default: 0
    }

}, { timestamps: true })


const Star = mongoose.model('Star', starSchema);

module.exports = { Star }

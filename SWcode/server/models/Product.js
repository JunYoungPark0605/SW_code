const mongoose = require('mongoose');
var ProductCounter = require('./ProductCounter');
const Schema = mongoose.Schema;

const productSchema = mongoose.Schema({
    writer: {
        type: Schema.Types.ObjectId,
        ref: 'User'
    },
    title: {
        type: String,
        maxlength: 50
    },
    description: {
        type: String,
    },
    callNumber: {
        type: String,
    },
    reviewlink: {
        type: String,
    },
    price: {
        type: Number,
        default: 0
    },
    images: {
        type: Array,
        default: []
    },
    sold: {
        type: Number,
        maxlength: 100,
        default: 0
    },

    Categories: {
        type: Number,
        default: 1
    },

    detailCategories: {
        type: Number,
        default: 0
    },

    spots: {
        type: Number,
        default: 1
    },
    
    views: {
        type: Number,
        default: 0
    },

    pCounter: {
        type: Number,
        default: 0
    }
}, { timestamps: true })

productSchema.index({
    title: 'text',
    description: 'text'
}, {
    weights: {
        title: 5,
        description: 1
    }
})

// ProductCounter.productCount++;
// ProductCounter.save();
// product.pCounter = ProductCounter.productCount;

const Product = mongoose.model('Product', productSchema);

module.exports = { Product }
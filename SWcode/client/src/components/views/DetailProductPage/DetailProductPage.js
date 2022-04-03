import React, { useEffect, useState } from 'react'
import axios from 'axios';
import ProductImage from './Sections/ProductImage';
import ProductInfo from './Sections/ProductInfo';
import {List, Avatar, Row, Col } from 'antd';
import Comments from './Sections/Comments'
import LikeDislikes from './Sections/LikeDislikes';
import Star from './Sections/Star';

function DetailProductPage(props) {

    const productId = props.match.params.productId

    const [Product, setProduct] = useState({})
    const [CommentLists, setCommentLists] = useState([])

    const productVariable = {
        productId: productId
    }

    useEffect(() => {
        axios.get(`/api/product/products_by_id?id=${productId}&type=single`)
            .then(response => {
                setProduct(response.data[0])
            })
            .catch(err => alert(err))

            axios.post('/api/comment/getComments', productVariable)
            .then(response => {
                if (response.data.success) {
                    console.log('response.data.comments',response.data.comments)
                    setCommentLists(response.data.comments)
                } else {
                    alert('Failed to get product Info')
                }
            })
    }, [])

    const updateComment = (newComment) => {
        setCommentLists(CommentLists.concat(newComment))
    }


    return (
        <div style={{ width: '100%', padding: '3rem 4rem' }}>

            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <h1>{Product.title}</h1>
            </div>

            <br />

            <Row gutter={[16, 16]} >
                <Col lg={12} sm={24}>
                    {/* ProductImage */}
                    <ProductImage detail={Product} />
                </Col>
                <Col lg={12} sm={24}>
                    {/* ProductInfo */}
                    <ProductInfo detail={Product} />
                </Col>
            </Row>
                <List.Item
                    style={{ justifyContent: 'center' }}
                    actions={[<LikeDislikes product productId={productId} userId={localStorage.getItem('userId')}  />]}
                >
                </List.Item>

                <Comments CommentLists={CommentLists} productId={productId} refreshFunction={updateComment} />



        </div>
    )
}

export default DetailProductPage

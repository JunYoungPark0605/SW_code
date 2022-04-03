import React, { useEffect, useState } from 'react'
import axios from 'axios';
import RequestInfo from './Section/RequestInfo';
import {List, Avatar, Row, Col } from 'antd';
import Comments from './Section/Comments';

function DetailRequestPage(props) {

    const requestId = props.match.params.requestId

    const [Request, setRequest] = useState({})
    const [CommentLists, setCommentLists] = useState([])

    const requestVariable = {
        requestId: requestId
    }

    useEffect(() => {
        axios.get(`/api/request/requests_by_id?id=${requestId}&type=single`)
            .then(response => {
                setRequest(response.data[0])
            })
            .catch(err => alert(err))

            axios.post('/api/comment/getComments', requestVariable)
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
                <h1>{Request.title}</h1>
            </div>

            <Row gutter={[16, 16]} >
                <Col lg={12} sm={24}>
                </Col>
                <Col lg={12} sm={24}>
                    {/* ProductInfo */}
                    <RequestInfo detail={Request} />
                </Col>

                <Comments CommentLists={CommentLists} requestId={requestId} refreshFunction={updateComment} />

            </Row>





        </div>
    )
}

export default DetailRequestPage

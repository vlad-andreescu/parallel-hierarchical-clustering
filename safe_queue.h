#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <class E>
class SafeUnboundedQueueCV {
    std::queue<E> elements;
    std::mutex lock;
    std::condition_variable not_empty;
public:
    SafeUnboundedQueueCV() {}
    void push(const E& element) {
        std::unique_lock<std::mutex> guard(lock);
        elements.push(element);
        not_empty.notify_all();
    }
    E pop() {
        std::unique_lock<std::mutex> guard(lock);
        while (elements.empty())
            not_empty.wait(guard);
        E el = elements.front();
        elements.pop();
        return el;
    }
};